import argparse
import itertools
import json
import os
import time
from datetime import datetime, timezone

from constants import DB_PATH, InsuranceType
from rag.rag import RAG
from rag.prepare_faq_for_rag import FAQDataExtractor
from rag.prepare_pdf_for_rag import PDFDataProcessor
from rag.prepare_policies_for_rag import PoliciesDataProcessor

CHECKPOINT_PATH = os.path.join(os.path.dirname(DB_PATH), "rag_checkpoint.json")


def faq_wrapper(json_path: str):
    generator = FAQDataExtractor(json_path)
    wrapper = generator.extract_faq_entries()
    for entry in wrapper:
        kwargs = dict()
        kwargs["chunk"] = entry["formatted_qa"]
        kwargs["insurance_type"] = InsuranceType.from_string(entry["topic"])
        kwargs["full_doc"] = entry["formatted_qa"]
        kwargs["url"] = entry["url"]
        kwargs["hyperlinks"] = entry["hyperlinks"]
        yield kwargs


def pdf_wrapper(json_path: str):
    extractor = PDFDataProcessor(json_path)
    generator = extractor.iter_all_segments()
    for entry in generator:
        kwargs = dict()
        kwargs["chunk"] = entry.page_chunk
        kwargs["insurance_type"] = InsuranceType.from_string(entry.topic)
        kwargs["full_doc"] = entry.entire_page_data
        kwargs["url"] = entry.url
        kwargs["page_index"] = entry.page_number
        yield kwargs


def html_wrapper(json_path: str):
    extractor = PoliciesDataProcessor(json_path)
    generator = extractor.iter_all_segments()
    for entry in generator:
        kwargs = dict()
        kwargs["chunk"] = entry.text
        kwargs["insurance_type"] = InsuranceType.from_string(entry.topic)
        kwargs["full_doc"] = entry.full_page_text
        kwargs["url"] = entry.url
        kwargs["hyperlinks"] = entry.hyperlinks
        yield kwargs


ALL_SOURCES = {
    "faq": ("FAQ", lambda: faq_wrapper("dataset/dataset-output/dataset/topics-faq.json")),
    "pdf": ("PDF", lambda: pdf_wrapper("dataset/dataset-output/dataset/topics-pdf-docling.json")),
    "blogs": ("Blogs", lambda: html_wrapper("dataset/dataset-output/dataset/topic-information-blogs-manual-html-parse.json")),
    "policies": ("Policies", lambda: html_wrapper("dataset/dataset-output/dataset/topics-policies-manual-html-parse.json")),
}


# --- Checkpoint helpers ---

def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _load_checkpoint(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != 1:
            return None
        return data
    except (json.JSONDecodeError, KeyError):
        return None


def _save_checkpoint(path: str, checkpoint: dict):
    checkpoint["updated_at"] = _now_iso()
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _create_fresh_checkpoint(source_keys: list[str], limit: int | None) -> dict:
    sources = {}
    for key in source_keys:
        sources[key] = {
            "status": "pending",
            "items_inserted": 0,
            "started_at": None,
            "completed_at": None,
            "elapsed_seconds": None,
        }
    return {
        "version": 1,
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
        "source_keys_requested": source_keys,
        "limit": limit,
        "sources": sources,
    }


# --- Main build logic ---

def rebuild_rag(limit: int | None = None, source_keys: list[str] | None = None, resume: bool = False):
    keys = source_keys or list(ALL_SOURCES.keys())

    # Checkpoint handling
    checkpoint = None
    if resume:
        checkpoint = _load_checkpoint(CHECKPOINT_PATH)
        if checkpoint is None:
            print("No valid checkpoint found. Starting fresh build.")
            resume = False
        elif checkpoint["source_keys_requested"] != keys:
            print(f"Checkpoint source keys {checkpoint['source_keys_requested']} "
                  f"don't match requested {keys}. Starting fresh build.")
            checkpoint = None
            resume = False
        elif checkpoint.get("limit") != limit:
            print(f"Checkpoint limit ({checkpoint.get('limit')}) doesn't match "
                  f"requested ({limit}). Starting fresh build.")
            checkpoint = None
            resume = False
        else:
            print(f"Resuming from checkpoint (updated {checkpoint['updated_at']})")

    if not resume:
        checkpoint = _create_fresh_checkpoint(keys, limit)

    # Create RAG instance
    reset_collection = not resume
    rag = RAG(reset_collection=reset_collection)

    if not resume:
        _save_checkpoint(CHECKPOINT_PATH, checkpoint)

    total_start = time.time()

    # Process each source
    for key in keys:
        src_state = checkpoint["sources"][key]

        if src_state["status"] == "completed":
            print(f"Skipping {key} (already completed: {src_state['items_inserted']} items)")
            continue

        name, gen_fn = ALL_SOURCES[key]
        gen = gen_fn()

        # Apply limit
        if limit:
            gen = itertools.islice(gen, limit)

        # Skip already-inserted items for partially completed source
        skip_count = src_state["items_inserted"]
        if skip_count > 0:
            print(f"Resuming {name}: skipping {skip_count} already-inserted items...")
            gen = itertools.islice(gen, skip_count, None)

        # Mark source as in_progress
        src_state["status"] = "in_progress"
        src_state["started_at"] = src_state["started_at"] or _now_iso()
        _save_checkpoint(CHECKPOINT_PATH, checkpoint)

        source_start = time.time()
        print(f"Loading {name}...")

        # Progress callback to save checkpoint after each batch
        def make_on_batch(source_key, already_done):
            def on_batch_inserted(batch_count, total_in_source):
                checkpoint["sources"][source_key]["items_inserted"] = already_done + total_in_source
                _save_checkpoint(CHECKPOINT_PATH, checkpoint)
            return on_batch_inserted

        try:
            inserted = rag.load_data_from_generator(
                gen,
                on_batch_inserted=make_on_batch(key, skip_count),
            )
        except Exception as e:
            source_elapsed = time.time() - source_start
            print(f"  ERROR in {name} after {source_elapsed:.1f}s: {e}")
            _save_checkpoint(CHECKPOINT_PATH, checkpoint)
            raise

        source_elapsed = time.time() - source_start
        src_state["status"] = "completed"
        src_state["items_inserted"] = skip_count + inserted
        src_state["completed_at"] = _now_iso()
        src_state["elapsed_seconds"] = round(source_elapsed, 2)
        _save_checkpoint(CHECKPOINT_PATH, checkpoint)

        print(f"  {name} done: {src_state['items_inserted']} items in {source_elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\nAll sources complete. Total time: {total_elapsed:.1f}s")
    print(f"Checkpoint saved at {CHECKPOINT_PATH}")

    return rag


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG vector DB")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max docs to ingest per source (for testing)")
    parser.add_argument("--sources", nargs="+", choices=list(ALL_SOURCES.keys()),
                        default=None, help="Which sources to include (default: all)")

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true",
                              help="Resume from last checkpoint")
    resume_group.add_argument("--fresh", action="store_true",
                              help="Force fresh build (ignore checkpoint)")

    args = parser.parse_args()

    # Auto-detect incomplete checkpoint if neither flag given
    if not args.fresh and not args.resume:
        cp = _load_checkpoint(CHECKPOINT_PATH)
        if cp and any(s["status"] != "completed" for s in cp["sources"].values()):
            print(f"Incomplete checkpoint found at {CHECKPOINT_PATH}")
            print("Use --resume to continue or --fresh to start over.")
            print("Defaulting to --fresh.\n")

    rebuild_rag(limit=args.limit, source_keys=args.sources, resume=args.resume)
