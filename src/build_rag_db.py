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


def rebuild_rag():
    rag = RAG(reset_collection=True)
    rag.load_data_from_generator(faq_wrapper("dataset/dataset-output/dataset/topics-faq.json"))
    rag.load_data_from_generator(pdf_wrapper("dataset/dataset-output/dataset/topics-pdf-docling.json"))
    rag.load_data_from_generator(html_wrapper("dataset/dataset-output/dataset/topic-information-blogs-manual-html-parse.json"))
    rag.load_data_from_generator(html_wrapper("dataset/dataset-output/dataset/topics-policies-manual-html-parse.json"))
    return rag

if __name__ == "__main__":
    rebuild_rag()
