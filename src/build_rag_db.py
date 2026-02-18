import argparse
import itertools
import json

from constants import InsuranceType
from rag.rag import RAG
from rag.prepare_faq_for_rag import FAQDataExtractor
from rag.prepare_pdf_for_rag import PDFDataProcessor
from rag.prepare_policies_for_rag import PoliciesDataProcessor


def faq_wrapper(json_path: str):
    generator = FAQDataExtractor(json_path)
    wrapper = generator.extract_faq_entries()
    for entry in wrapper:
        kwargs = dict()
        kwargs["chunk"] = entry["question"]
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


def rebuild_rag(limit: int | None = None, source_keys: list[str] | None = None):
    rag = RAG(reset_collection=True)

    keys = source_keys or list(ALL_SOURCES.keys())
    for key in keys:
        name, gen_fn = ALL_SOURCES[key]
        gen = gen_fn()
        if limit:
            gen = itertools.islice(gen, limit)
        print(f"Loading {name}...")
        rag.load_data_from_generator(gen)
        print(f"  {name} done.")

    return rag

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG vector DB")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max docs to ingest per source (for testing)")
    parser.add_argument("--sources", nargs="+", choices=list(ALL_SOURCES.keys()),
                        default=None, help="Which sources to include (default: all)")
    args = parser.parse_args()
    rebuild_rag(limit=args.limit, source_keys=args.sources)
