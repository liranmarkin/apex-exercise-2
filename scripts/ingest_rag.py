"""Ingest all RAG data dumps into Milvus vector database."""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from constants import InsuranceType, SourceType
from rag.rag import RAG

DUMPS_DIR = os.path.join(os.path.dirname(__file__), "../dataset/dataset-output/rag_dumps")

TOPIC_TO_INSURANCE_TYPE = {
    "travel": InsuranceType.TRAVEL,
    "health": InsuranceType.HEALTH,
    "car": InsuranceType.CAR,
    "apartment": InsuranceType.APARTMENT,
    "life": InsuranceType.LIFE,
    "business": InsuranceType.BUSINESS,
    "dental": InsuranceType.DENTAL,
    "mortgage": InsuranceType.MORTGAGE,
}

MAX_FULL_DOC_LEN = 10000
MAX_PAGE_INDEX = 127


def _truncate(text: str, max_len: int) -> str:
    return text[:max_len] if text else ""


def _parse_hyperlinks(raw) -> dict:
    """Normalize hyperlinks from various formats to dict[str, str]."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        result = {}
        for item in raw:
            if isinstance(item, dict) and "text" in item and "url" in item:
                result[item["text"]] = item["url"]
            elif isinstance(item, dict) and "href" in item:
                result[item.get("text", "")] = item["href"]
        return result
    return {}


def pdf_generator(data: dict):
    for seg in data["segments"]:
        topic = seg.get("topic", "").lower()
        insurance_type = TOPIC_TO_INSURANCE_TYPE.get(topic)
        if not insurance_type:
            continue

        chunk = seg.get("page_chunk", "")
        full_doc = seg.get("entire_page_data", chunk)
        if not chunk:
            continue

        page = seg.get("page_number", -1)
        if page is None or page < 0:
            page = -1
        page = min(page, MAX_PAGE_INDEX)

        yield {
            "chunk": chunk,
            "insurance_type": insurance_type,
            "full_doc": _truncate(full_doc, MAX_FULL_DOC_LEN),
            "url": seg.get("url", ""),
            "page_index": page,
            "source_type": SourceType.PDF,
        }


def faq_generator(data: dict):
    for entry in data["entries"]:
        topic = entry.get("topic", "").lower()
        insurance_type = TOPIC_TO_INSURANCE_TYPE.get(topic)
        if not insurance_type:
            continue

        chunk = entry.get("formatted_qa", "")
        if not chunk:
            continue

        yield {
            "chunk": chunk,
            "insurance_type": insurance_type,
            "full_doc": _truncate(chunk, MAX_FULL_DOC_LEN),
            "url": entry.get("url", ""),
            "hyperlinks": _parse_hyperlinks(entry.get("hyperlinks", {})),
            "source_type": SourceType.FAQ,
        }


def segment_generator(data: dict, source_type: SourceType = SourceType.POLICY):
    """Generator for policies and blogs dumps (same format)."""
    for seg in data["segments"]:
        topic = seg.get("topic", "").lower()
        insurance_type = TOPIC_TO_INSURANCE_TYPE.get(topic)
        if not insurance_type:
            continue

        chunk = seg.get("text", "")
        full_doc = seg.get("full_page_text", chunk)
        if not chunk:
            continue

        yield {
            "chunk": chunk,
            "insurance_type": insurance_type,
            "full_doc": _truncate(full_doc, MAX_FULL_DOC_LEN),
            "url": seg.get("url", ""),
            "hyperlinks": _parse_hyperlinks(seg.get("hyperlinks", {})),
            "source_type": source_type,
        }


DUMP_CONFIG = [
    ("pdf_dump.json", lambda data: pdf_generator(data)),
    ("faq_dump.json", lambda data: faq_generator(data)),
    ("policies_dump.json", lambda data: segment_generator(data, SourceType.POLICY)),
    ("blogs_dump.json", lambda data: segment_generator(data, SourceType.BLOG)),
]


def ingest(limit_per_dump: int | None = None):
    print("Initializing RAG with reset_collection=True ...")
    rag = RAG(reset_collection=True)

    total_inserted = 0
    for filename, gen_fn in DUMP_CONFIG:
        path = os.path.join(DUMPS_DIR, filename)
        print(f"\nLoading {filename} ...")
        with open(path) as f:
            data = json.load(f)

        count = 0
        start = time.time()
        for doc in gen_fn(data):
            rag.insert_doc(**doc)
            count += 1
            total_inserted += 1
            if count % 100 == 0:
                elapsed = time.time() - start
                rate = count / elapsed
                print(f"  {filename}: {count} docs ingested ({rate:.1f} docs/s)")
            if limit_per_dump and count >= limit_per_dump:
                print(f"  Reached limit of {limit_per_dump} for {filename}")
                break

        elapsed = time.time() - start
        print(f"  Done: {count} docs from {filename} in {elapsed:.1f}s")

    print(f"\nTotal: {total_inserted} docs ingested into Milvus")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest RAG data dumps into Milvus")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max docs per dump file (for testing)")
    args = parser.parse_args()
    ingest(limit_per_dump=args.limit)
