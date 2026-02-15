from prepare_policies_for_rag import PoliciesDataProcessor
from prepare_pdf_for_rag import PDFDataProcessor
from prepare_faq_for_rag import FAQDataExtractor

############## FAQ ##############
extractor = FAQDataExtractor('dataset/topics-faq.json')
entries = extractor.extract_faq_entries()
documents_faq = []
# For each entry, create a document for your RAG system
for entry in entries:
    document = {
        'content': entry['formatted_qa'],      # Main text for embedding
        'metadata': {
            'topic': entry['topic'],
            'url': entry['url'],
            'question': entry['question'],
            'hyperlinks': entry['hyperlinks']
        }
    }
    documents_faq.append(document)
#print(documents_faq)

############## PDF ##############
processor = PDFDataProcessor('dataset/topics-pdf-docling.json')

documents_pdf = []
for segment in processor.iter_all_segments():
    # Skip empty segments
    if not segment.page_chunk.strip():
        continue
    
    # Create document for RAG
    doc = {
        'id': f"{segment.url}#page{segment.page_number}#{hash(segment.page_chunk)}#{segment.chunk_type}",
        'text': segment.page_chunk,
        'metadata': {
            'topic': segment.topic,
            'url': segment.url,
            'page_number': segment.page_number,
            'chunk_type': segment.chunk_type,
            'content_type': segment.content_type,
            'full_page_context': segment.entire_page_data,
            'word_count': len(segment.page_chunk.split()),
            'source_type': 'pdf',
            'is_table': segment.content_type == 'table',
            'is_table_row': segment.chunk_type.startswith('table_row_')
        }
    }
    documents_pdf.append(doc)
#print(documents_pdf)

############## Blogs ############
processor = PoliciesDataProcessor('dataset/topic-information-blogs-manual-html-parse.json')
documents_blogs = []
for segment in processor.iter_all_segments():
    # Skip empty segments
    if not segment.text.strip():
        continue
    
    # Create document for RAG
    doc = {
        'id': f"{segment.url}#{hash(segment.text)}#{segment.chunk_type}",
        'text': segment.text,
        'metadata': {
            'topic': segment.topic,
            'url': segment.url,
            'chunk_type': segment.chunk_type,
            'has_links': len(segment.hyperlinks) > 0,
            'full_context': segment.full_page_text,
            'word_count': len(segment.text.split())
        }
    }
    documents_blogs.append(doc)
#print(documents_blogs)

############## Policies ############
processor = PoliciesDataProcessor('dataset/topics-policies-manual-html-parse.json')
documents_policies = []
for segment in processor.iter_all_segments():
    # Skip empty segments
    if not segment.text.strip():
        continue
    
    # Create document for RAG
    doc = {
        'id': f"{segment.url}#{hash(segment.text)}#{segment.chunk_type}",
        'text': segment.text,
        'metadata': {
            'topic': segment.topic,
            'url': segment.url,
            'chunk_type': segment.chunk_type,
            'has_links': len(segment.hyperlinks) > 0,
            'full_context': segment.full_page_text,
            'word_count': len(segment.text.split())
        }
    }
    documents_policies.append(doc)
#print(documents_policies)