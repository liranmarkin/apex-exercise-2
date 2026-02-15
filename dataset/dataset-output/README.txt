# Parse information blogs
python3 scripts/parse_html_files.py ../dataset-parse/data/topics-information-blogs dataset/topic-information-blogs-manual-html-parse.json

# Parse policies
python3 scripts/parse_html_files.py ../dataset-parse/data/topics-policies dataset/topics-policies-manual-html-parse.jso

# Parse PDFs
python3 scripts/parse_docling_json.py ../dataset/dataset-parse/data/pdfs/ dataset/topics-pdf-docling.json _db_pdf.csv





# prepare for RAG
python3 scripts/prep-rag/prepare_faq_for_rag.py dataset/topics-faq.json
python3 scripts/prep-rag/prepare_policies_for_rag.py dataset/topics-policies-manual-html-parse.json --demo --chunk-threshold 30
python3 scripts/prep-rag/prepare_policies_for_rag.py dataset/topic-information-blogs-manual-html-parse.json --demo --chunk-threshold 30





# Run all for RAG
python3 scripts/prep-rag/dump_all_datasets.py --all --blogs dataset/topic-information-blogs-manual-html-parse.json --policies dataset/topics-policies-manual-html-parse.json --pdf dataset/topics-pdf-docling.json --faq dataset/topics-faq.json --output-dir rag_dumps --chunk-threshold 50