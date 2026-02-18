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


def rebuild_rag(limit: int | None = None):
    rag = RAG(reset_collection=True)

    sources = [
        ("FAQ", faq_wrapper("dataset/dataset-output/dataset/topics-faq.json")),
        ("PDF", pdf_wrapper("dataset/dataset-output/dataset/topics-pdf-docling.json")),
        ("Blogs", html_wrapper("dataset/dataset-output/dataset/topic-information-blogs-manual-html-parse.json")),
        ("Policies", html_wrapper("dataset/dataset-output/dataset/topics-policies-manual-html-parse.json")),
    ]

    for name, gen in sources:
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
    args = parser.parse_args()
    rebuild_rag(limit=args.limit)
