# RAG Data Preparation Scripts

This directory contains scripts to prepare different data sources for RAG (Retrieval-Augmented Generation) systems.

## Quick Start: Dump All Datasets

Use the unified dump script to export all datasets at once:

```bash
python3 scripts/prep-rag/dump_all_datasets.py --all \
  --policies dataset/topics-policies-manual-html-parse.json \
  --pdf dataset/topics-pdf-docling.json \
  --faq dataset/topics-faq.json \
  --output-dir rag_dumps
```

This creates 4 JSON files:
- `policies_dump.json` - 1,260 segments (~6 MB)
- `pdf_dump.json` - 49,388 segments (~219 MB)
- `faq_dump.json` - 203 entries (~222 KB)
- `dump_summary.json` - Combined statistics

📖 [Full Dump Script Documentation](README_DUMP.md)

---

## Available Processors

### 1. Policies Processor (HTML Data)
**File**: `prepare_policies_for_rag.py`  
**Input**: `dataset/topics-policies-manual-html-parse.json`  
**Source**: HTML web pages with manual parsing

**Features**:
- Filters only `strHTML` fields
- URL normalization (removes `/index`)
- Smart chunking with overlap
- Returns: topic, text, full_page_text, hyperlinks, url, chunk_type

**Quick Start**:
```bash
# Show statistics
python3 scripts/prep-rag/prepare_policies_for_rag.py dataset/topics-policies-manual-html-parse.json

# Run demo
python3 scripts/prep-rag/prepare_policies_for_rag.py dataset/topics-policies-manual-html-parse.json --demo
```

📖 [Full Documentation](README_POLICIES_RAG.md)

---

### 2. PDF Processor (Docling Data)
**File**: `prepare_pdf_for_rag.py`  
**Input**: `dataset/topics-pdf-docling.json`  
**Source**: PDF documents parsed with Docling

**Features**:
- Page-level organization
- URL normalization
- Smart chunking with overlap
- Returns: topic, page_chunk, entire_page_data, page_number, url, chunk_type

**Quick Start**:
```bash
# Show statistics
python3 scripts/prep-rag/prepare_pdf_for_rag.py dataset/topics-pdf-docling.json

# Run demo
python3 scripts/prep-rag/prepare_pdf_for_rag.py dataset/topics-pdf-docling.json --demo
```

📖 [Full Documentation](README_PDF_RAG.md)

---

### 3. FAQ Processor
**File**: `prepare_faq_for_rag.py`  
**Input**: `dataset/topics-faq.json`  
**Source**: FAQ data

**Features**:
- Question-answer pairs
- Topic categorization
- URL tracking

**Quick Start**:
```bash
python3 scripts/prep-rag/prepare_faq_for_rag.py dataset/topics-faq.json
```

📖 [Full Documentation](README_FAQ_RAG.md)

---

## Common Features

All processors support:

✅ **Smart Chunking**: Automatically creates overlapping chunks for longer texts
- Full text (always included)
- 2-sentence chunks (chunk_1-2, chunk_2-3, etc.)
- 3-sentence chunks (chunk_1-3, chunk_2-4, etc.)
- 4-sentence chunks (chunk_1-4, chunk_2-5, etc.)

✅ **Configurable Threshold**: Set word count threshold for chunking
```bash
--chunk-threshold 40  # Default is 50
```

✅ **Topic Filtering**: Get segments by topic
```python
segments = processor.get_segments_by_topic('travel')
```

✅ **URL Filtering**: Get segments by URL
```python
segments = processor.get_segments_by_url('https://example.com/page')
```

✅ **Statistics**: Get dataset overview
```python
stats = processor.get_statistics()
```

✅ **JSON Export**: Export segments to JSON format
```python
json_data = segment.to_dict()
```

---

## Dataset Overview

| Dataset | Files | Segments | Topics | URLs/Pages |
|---------|-------|----------|--------|------------|
| Policies (HTML) | 87 | 1,101 | 8 | 82 URLs |
| PDF (Docling) | 178 | 56,998 | 7 | 2,845 pages |
| FAQ | TBD | TBD | TBD | TBD |

---

## Usage Patterns

### Pattern 1: Process All Data Sources

```python
from prepare_policies_for_rag import PoliciesDataProcessor
from prepare_pdf_for_rag import PDFDataProcessor
from prepare_faq_for_rag import FAQDataProcessor

# Initialize all processors
policies = PoliciesDataProcessor('dataset/topics-policies-manual-html-parse.json')
pdfs = PDFDataProcessor('dataset/topics-pdf-docling.json')
faqs = FAQDataProcessor('dataset/topics-faq.json')

# Combine all segments
all_segments = []
all_segments.extend(list(policies.iter_all_segments()))
all_segments.extend(list(pdfs.iter_all_segments()))
all_segments.extend(list(faqs.iter_all_segments()))

print(f"Total segments: {len(all_segments)}")
```

### Pattern 2: Filter by Topic Across Sources

```python
topic = 'travel'

travel_policies = policies.get_segments_by_topic(topic)
travel_pdfs = pdfs.get_segments_by_topic(topic)
travel_faqs = faqs.get_segments_by_topic(topic)

print(f"Travel segments:")
print(f"  Policies: {len(travel_policies)}")
print(f"  PDFs: {len(travel_pdfs)}")
print(f"  FAQs: {len(travel_faqs)}")
```

### Pattern 3: Insert into Vector Database

```python
# Example with any processor
processor = PoliciesDataProcessor('dataset/topics-policies-manual-html-parse.json')

documents = []
for segment in processor.iter_all_segments():
    if not segment.text.strip():
        continue
    
    doc = {
        'id': f"{segment.url}#{hash(segment.text)}#{segment.chunk_type}",
        'text': segment.text,
        'metadata': {
            'topic': segment.topic,
            'url': segment.url,
            'chunk_type': segment.chunk_type,
            'source': 'policies'
        }
    }
    documents.append(doc)

# Insert into your vector database
# vector_db.insert(documents)
```

---

## Chunking Strategy

All processors use the same chunking strategy:

1. **Threshold Check**: If text has ≤ threshold words, no chunking
2. **Sentence Splitting**: Split text into sentences
3. **Overlapping Chunks**: Create multiple chunk sizes with overlap
4. **Full Text Preservation**: Always include the complete original text

**Benefits**:
- Better semantic matching for queries
- Reduced boundary issues
- Multiple granularities for retrieval
- Context preservation

**Example**:
```
Original: "Sentence 1. Sentence 2. Sentence 3. Sentence 4."

Chunks created:
- full: "Sentence 1. Sentence 2. Sentence 3. Sentence 4."
- chunk_1-2: "Sentence 1. Sentence 2."
- chunk_2-3: "Sentence 2. Sentence 3."
- chunk_3-4: "Sentence 3. Sentence 4."
- chunk_1-3: "Sentence 1. Sentence 2. Sentence 3."
- chunk_2-4: "Sentence 2. Sentence 3. Sentence 4."
```

---

## Best Practices

1. **Choose appropriate threshold**: 
   - Lower (30-40) for shorter, focused chunks
   - Higher (50-70) for more context per chunk

2. **Filter by chunk_type**:
   - Use `chunk_type == "full"` for complete texts
   - Use all chunks for maximum retrieval flexibility

3. **Combine metadata**:
   - Include topic, URL, page numbers in vector DB metadata
   - Enables filtered retrieval and source attribution

4. **Test retrieval**:
   - Experiment with different chunk strategies
   - Measure retrieval quality with your queries

5. **Monitor statistics**:
   - Check how many chunks are created
   - Adjust threshold based on your needs

---

## Contributing

To add a new processor:

1. Create `prepare_<source>_for_rag.py`
2. Implement similar API (get_segment, iter_all_segments, get_statistics)
3. Add chunking support using `_create_chunks()`
4. Create README_<SOURCE>_RAG.md
5. Update this main README

---

## License

[Your License Here]
