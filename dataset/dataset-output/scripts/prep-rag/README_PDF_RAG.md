# PDF Data Processor for RAG System

This script processes `topics-pdf-docling.json` (PDF documents parsed with Docling) and provides API functions to extract structured data for RAG (Retrieval-Augmented Generation) systems.

## Key Features

- **PDF-specific processing**: Extracts text content from parsed PDF documents
- **Arabic PDF filtering**: Automatically filters out PDFs with 'ערבית' in filename
- **Text normalization**: Cleans unicode artifacts, extra whitespace, and special characters
- **Table support**: Processes tables with smart chunking (headers + individual rows)
- **Page-level organization**: Groups content by page number
- **URL normalization**: Removes `/index` suffix from URLs
- **Smart chunking**: Automatically creates overlapping chunks for longer texts
- **PDFSegment API**: Returns structured data including:
  - Topic
  - Page chunk (text segment or table row)
  - Entire page data (all text and tables from the same page)
  - Page number
  - Source URL
  - Chunk type (full, chunk_1-2, table_row_1, etc.)
  - Content type (text or table)

## Text Normalization

All text is automatically normalized to improve quality:

1. **Unicode artifact removal**: Removes `/uniXXXX` patterns (e.g., `/uniFEEE`)
2. **Unicode space normalization**: Converts various unicode spaces to regular spaces
   - Non-breaking spaces (\u00A0)
   - Em/en spaces (\u2000-\u2009)
   - Zero-width spaces (\u200B, \uFEFF)
   - And more...
3. **Whitespace collapsing**: Multiple spaces/tabs collapsed to single space
4. **Line break cleanup**: Removes excessive newlines (max 2 for paragraph breaks)
5. **Trim whitespace**: Removes leading/trailing spaces from lines

## Table Processing

Tables are processed with special chunking strategy:
1. **Full table**: Complete table with all rows (chunk_type: "full")
2. **Row chunks**: Headers + individual data row (chunk_type: "table_row_1", "table_row_2", etc.)

This allows RAG systems to:
- Retrieve entire tables for context
- Match specific rows for precise answers
- Always include headers for understanding

**Example table chunk**:
```
Headers: השתתפות עצמית | האחריות גבול | הכיסוי
Row:     ה.ע ללא | $5,000,000 | רפואיות הוצאות
```

## Chunking Strategy

When text exceeds a word count threshold (default: 50 words), the processor creates:
1. **Full text**: The complete original text (chunk_type: "full")
2. **2-sentence chunks**: Overlapping pairs (chunk_1-2, chunk_2-3, etc.)
3. **3-sentence chunks**: Overlapping triplets (chunk_1-3, chunk_2-4, etc.)
4. **4-sentence chunks**: Overlapping quads (chunk_1-4, chunk_2-5, etc.)

This provides multiple granularities for better RAG retrieval.

## Usage

### Command Line

```bash
# Show statistics (default threshold: 50 words)
python3 scripts/prep-rag/prepare_pdf_for_rag.py dataset/topics-pdf-docling.json

# Show statistics with custom threshold
python3 scripts/prep-rag/prepare_pdf_for_rag.py dataset/topics-pdf-docling.json --chunk-threshold 30

# Run demonstration
python3 scripts/prep-rag/prepare_pdf_for_rag.py dataset/topics-pdf-docling.json --demo

# Run demonstration with custom threshold
python3 scripts/prep-rag/prepare_pdf_for_rag.py dataset/topics-pdf-docling.json --demo --chunk-threshold 40
```

### Python API

```python
from prepare_pdf_for_rag import PDFDataProcessor

# Initialize processor with custom chunk threshold
processor = PDFDataProcessor(
    'dataset/topics-pdf-docling.json',
    chunk_threshold=50  # Words threshold for chunking
)

# Get specific PDF segment (returns list of segments if chunked)
segments = processor.get_pdf_segment(file_index=0, content_index=0)
if segments:
    for segment in segments:
        print(f"Chunk type: {segment.chunk_type}")
        print(f"Topic: {segment.topic}")
        print(f"Page chunk: {segment.page_chunk}")
        print(f"Page number: {segment.page_number}")
        print(f"URL: {segment.url}")
        print(f"Entire page: {segment.entire_page_data}")

# Iterate through all segments (includes all chunks)
for segment in processor.iter_all_segments():
    if segment.chunk_type == "full":
        print(f"Page {segment.page_number}: {segment.page_chunk[:50]}...")
    else:
        print(f"Chunk {segment.chunk_type}: {segment.page_chunk[:50]}...")

# Get only full segments (no chunks) by topic
travel_segments = processor.get_segments_by_topic('travel')
full_segments = [s for s in travel_segments if s.chunk_type == "full"]
print(f"Found {len(full_segments)} full travel segments")

# Get segments by URL (normalized)
url_segments = processor.get_segments_by_url('https://media.harel-group.co.il/media/of1evuu4/הודעה-על-הגדרת-ספורט-אתגרי.pdf')

# Get segments from a specific page
page_segments = processor.get_segments_by_page(
    'https://media.harel-group.co.il/media/of1evuu4/הודעה-על-הגדרת-ספורט-אתגרי.pdf',
    page_number=1
)

# Filter by content type
text_segments = [s for s in page_segments if s.content_type == 'text']
table_segments = [s for s in page_segments if s.content_type == 'table']

# Get only table row chunks (not full tables)
table_rows = [s for s in table_segments if s.chunk_type.startswith('table_row_')]

# Get statistics
stats = processor.get_statistics()
print(f"Total segments: {stats['total_segments']}")
print(f"Text segments: {stats['text_segments']}")
print(f"Table segments: {stats['table_segments']}")
print(f"Table row chunks: {stats['table_row_chunks']}")
print(f"Total chunks (including overlaps): {stats['total_chunks_including_overlaps']}")
print(f"Unique pages: {stats['unique_pages']}")
print(f"Topics: {stats['topics_list']}")

# Export to JSON
segment_dict = segments[0].to_dict()
import json
json_output = json.dumps(segment_dict, ensure_ascii=False, indent=2)
```

## Data Structure

### PDFSegment Object

```python
@dataclass
class PDFSegment:
    topic: str                  # Topic category (e.g., 'travel', 'health')
    page_chunk: str             # Text content or table row (full or chunk)
    entire_page_data: str       # All text and tables from the same page
    page_number: int            # Page number in the PDF
    url: str                    # Normalized source URL (no /index suffix)
    chunk_type: str             # "full", "chunk_1-2", "table_row_1", etc.
    content_type: str           # "text" or "table"
```

### Example Output

**Text segment**:
```json
{
  "topic": "travel",
  "page_chunk": "ספורט אתגרי חובבני כולל ענפי ספורט...",
  "entire_page_data": "ספורט אתגרי חובבני כולל...\n\n[TABLE]\nהשתתפות עצמית | האחריות גבול...",
  "page_number": 1,
  "url": "https://media.harel-group.co.il/media/of1evuu4/הודעה-על-הגדרת-ספורט-אתגרי.pdf",
  "chunk_type": "full",
  "content_type": "text"
}
```

**Table row chunk**:
```json
{
  "topic": "travel",
  "page_chunk": "השתתפות עצמית | האחריות גבול | הכיסוי\nה.ע ללא | $5,000,000 | רפואיות הוצאות",
  "entire_page_data": "[TABLE]\nהשתתפות עצמית | האחריות גבול | הכיסוי\n...",
  "page_number": 1,
  "url": "https://media.harel-group.co.il/media/3q5m1jde/גבולות-אחריות-דרכון-first-class.pdf",
  "chunk_type": "table_row_1",
  "content_type": "table"
}
```

## Dataset Statistics

With default settings (50-word threshold) and Arabic PDFs filtered:
- Total files: 178 (2 Arabic PDFs filtered out)
- Total segments: 38,956
  - Text segments: 38,712
  - Table segments: 244
- Table row chunks: 1,870 (headers + individual rows)
- Total chunks (including overlaps): 46,007
- Additional chunks created: 7,051
- Unique topics: 7 (apartment, business, car, health, life, mortgage, travel)
- Unique URLs: 164
- Unique pages: 2,663

## Integration with RAG Systems

This processor is designed to prepare PDF data for RAG systems. Each segment can be:

1. **Embedded**: Use the `page_chunk` field for vector embeddings
2. **Contextualized**: Include `entire_page_data` for broader page context
3. **Categorized**: Filter by `topic` for domain-specific retrieval
4. **Sourced**: Track origin with `url` and `page_number`
5. **Multi-granular**: Use `chunk_type` to retrieve at different levels

## Example RAG Integration

```python
# Prepare data for vector database
processor = PDFDataProcessor('dataset/topics-pdf-docling.json')

documents = []
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
    documents.append(doc)

# Strategy 1: Insert all chunks for maximum retrieval flexibility
# This gives you full texts + overlapping chunks + table rows

# Strategy 2: Insert only full texts and full tables
full_docs = [d for d in documents if d['metadata']['chunk_type'] == 'full']

# Strategy 3: Insert full texts + table rows (no text chunks)
full_and_table_rows = [d for d in documents 
                       if d['metadata']['chunk_type'] == 'full' 
                       or d['metadata']['is_table_row']]

# Strategy 4: Separate text and table documents for different retrieval strategies
text_docs = [d for d in documents if d['metadata']['content_type'] == 'text']
table_docs = [d for d in documents if d['metadata']['content_type'] == 'table']

# Now insert documents into your vector database
# e.g., Pinecone, Weaviate, ChromaDB, etc.
```

## PDF-Specific Features

1. **Page-level context**: Each segment includes all text and tables from its page
2. **Page number tracking**: Easy to reference specific pages in source PDFs
3. **Multi-page document support**: Handles PDFs with multiple pages
4. **Docling compatibility**: Works with JSON output from Docling PDF parser
5. **Table processing**: Extracts tables with headers and creates row-level chunks
6. **Mixed content**: Handles pages with both text and tables

## Chunking Benefits for RAG

1. **Better matching**: Smaller chunks can match user queries more precisely
2. **Context preservation**: Full text and entire page data always available
3. **Overlap reduces boundary issues**: Important information at chunk boundaries is captured
4. **Multiple retrieval options**: Can retrieve at different granularities
5. **Improved relevance**: Shorter chunks often have higher semantic coherence
6. **Page-aware retrieval**: Can retrieve all content from a specific page
7. **Table-aware retrieval**: Can match specific table rows while preserving full table context
8. **Header preservation**: Table chunks always include headers for understanding

## Comparison with HTML Processor

| Feature | PDF Processor | HTML Processor |
|---------|--------------|----------------|
| Source | PDF documents | HTML web pages |
| Organization | By page number | By URL |
| Context field | `entire_page_data` | `full_page_text` |
| Text field | `page_chunk` | `text` |
| Additional metadata | `page_number` | `hyperlinks` |
| Typical use case | Policy documents, forms | Web content, FAQs |
