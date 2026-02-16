# PDF Data Processor for RAG System

This script processes `topics-pdf-docling.json` (PDF documents parsed with Docling) and provides API functions to extract structured data for RAG (Retrieval-Augmented Generation) systems.

## Key Features

- **PDF-specific processing**: Extracts text content from parsed PDF documents
- **Arabic PDF filtering**: Automatically filters out PDFs with 'ערבית' in filename
- **Text normalization**: Cleans unicode artifacts, extra whitespace, and special characters
- **Page-level processing**: Extracts full page text first, then chunks pages (not individual segments)
- **Table support with context**: Processes tables with:
  - 2 context chunks before each table (surrounding text)
  - Full table chunk
  - Individual row chunks formatted as "header is value" pairs
- **Cross-page context**: For boundary chunks, includes context from adjacent pages
- **URL normalization**: Removes `/index` suffix from URLs
- **Smart chunking**: Automatically creates overlapping chunks for longer texts
- **PDFSegment API**: Returns structured data including:
  - Topic
  - Page chunk (text segment or table row)
  - Entire page data (may include adjacent pages for boundary chunks)
  - Page number
  - Source URL
  - Chunk type (full, chunk_1-2, table_row_1, table_context_1, etc.)
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

Tables are processed with enhanced chunking strategy:
1. **Context chunk 1**: Surrounding text before the table (chunk_type: "table_context_1")
2. **Context chunk 2**: Surrounding text + table preview (chunk_type: "table_context_2")
3. **Full table**: Complete table with all rows (chunk_type: "full")
4. **Row chunks**: Individual data rows formatted as "header is value" pairs (chunk_type: "table_row_1", "table_row_2", etc.)

This allows RAG systems to:
- Understand table context from surrounding text
- Retrieve entire tables for context
- Match specific rows for precise answers
- Always have headers for understanding

**Example table row chunk (new format)**:
```
השתתפות עצמית is ה.ע ללא, האחריות גבול is $5,000,000, הכיסוי is רפואיות הוצאות
```

**Old format** (no longer used):
```
השתתפות עצמית | האחריות גבול | הכיסוי
ה.ע ללא | $5,000,000 | רפואיות הוצאות
```

## Chunking Strategy

### Text Chunking (Page-Level)
1. **Full page text extraction**: All text items on a page are concatenated first
2. **Page-level chunking**: If page text exceeds threshold (default: 50 words):
   - Full page text (chunk_type: "full")
   - Overlapping 2-sentence chunks (chunk_type: "chunk_1-2", "chunk_2-3", etc.)
   - Overlapping 3-sentence chunks (chunk_type: "chunk_1-3", "chunk_2-4", etc.)
   - Overlapping 4-sentence chunks (chunk_type: "chunk_1-4", "chunk_2-5", etc.)
3. **Cross-page context**: For non-full chunks, `entire_page_data` includes adjacent pages

This provides multiple granularities for better RAG retrieval while maintaining page-level coherence.

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

# Get all segments for a specific page (NEW API - changed from content_index to page_number)
segments = processor.get_pdf_segment(file_index=0, page_number=1)
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
    elif segment.chunk_type.startswith("table_context_"):
        print(f"Table context: {segment.page_chunk[:50]}...")
    elif segment.chunk_type.startswith("table_row_"):
        print(f"Table row: {segment.page_chunk[:50]}...")
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

# Get only table row chunks (not full tables or context)
table_rows = [s for s in table_segments if s.chunk_type.startswith('table_row_')]

# Get table context chunks
table_contexts = [s for s in table_segments if s.chunk_type.startswith('table_context_')]

# Get statistics
stats = processor.get_statistics()
print(f"Total segments: {stats['total_segments']}")
print(f"Text segments: {stats['text_segments']}")
print(f"Table segments: {stats['table_segments']}")
print(f"Table context chunks: {stats['table_context_chunks']}")
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
    entire_page_data: str       # Page context (may include adjacent pages for boundary chunks)
    page_number: int            # Page number in the PDF
    url: str                    # Normalized source URL (no /index suffix)
    chunk_type: str             # "full", "chunk_1-2", "table_row_1", "table_context_1", etc.
    content_type: str           # "text" or "table"
```

### Example Output

**Text segment (full page)**:
```json
{
  "topic": "travel",
  "page_chunk": "ספורט אתגרי חובבני כולל ענפי ספורט...",
  "entire_page_data": "[CURRENT PAGE 1]\nספורט אתגרי חובבני כולל...\n\n[TABLE]\nהשתתפות עצמית | האחריות גבול...",
  "page_number": 1,
  "url": "https://media.harel-group.co.il/media/of1evuu4/הודעה-על-הגדרת-ספורט-אתגרי.pdf",
  "chunk_type": "full",
  "content_type": "text"
}
```

**Text segment (boundary chunk with cross-page context)**:
```json
{
  "topic": "travel",
  "page_chunk": "ספורט אתגרי חובבני כולל...",
  "entire_page_data": "[PREVIOUS PAGE 1]\n...\n\n[CURRENT PAGE 2]\n...\n\n[NEXT PAGE 3]\n...",
  "page_number": 2,
  "url": "https://media.harel-group.co.il/media/of1evuu4/הודעה-על-הגדרת-ספורט-אתגרי.pdf",
  "chunk_type": "chunk_1-2",
  "content_type": "text"
}
```

**Table context chunk**:
```json
{
  "topic": "travel",
  "page_chunk": "איתור וחילוץ. בכל מקרה של סתירה...",
  "entire_page_data": "[TABLE]\nהשתתפות עצמית | האחריות גבול | הכיסוי\n...",
  "page_number": 1,
  "url": "https://media.harel-group.co.il/media/3q5m1jde/גבולות-אחריות-דרכון-first-class.pdf",
  "chunk_type": "table_context_1",
  "content_type": "table"
}
```

**Table row chunk (new format)**:
```json
{
  "topic": "travel",
  "page_chunk": "השתתפות עצמית is ה.ע ללא, האחריות גבול is $5,000,000, הכיסוי is רפואיות הוצאות",
  "entire_page_data": "[TABLE]\nהשתתפות עצמית | האחריות גבול | הכיסוי\n...",
  "page_number": 1,
  "url": "https://media.harel-group.co.il/media/3q5m1jde/גבולות-אחריות-דרכון-first-class.pdf",
  "chunk_type": "table_row_1",
  "content_type": "table"
}
```

## Dataset Statistics

With default settings (50-word threshold) and Arabic PDFs filtered:
- Total files: 176 (2 Arabic PDFs filtered out)
- Total segments: 2,948
  - Text segments: 2,704
  - Table segments: 244
- Table context chunks: 488 (2 per table)
- Table row chunks: 1,870 (headers + individual rows)
- Total chunks (including overlaps): 95,597
- Additional chunks created: 92,649
- Unique topics: 7 (apartment, business, car, health, life, mortgage, travel)
- Unique URLs: 164
- Unique pages: 2,663

## Integration with RAG Systems

This processor is designed to prepare PDF data for RAG systems. Each segment can be:

1. **Embedded**: Use the `page_chunk` field for vector embeddings
2. **Contextualized**: Include `entire_page_data` for broader page context (may include adjacent pages)
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
            'is_table_row': segment.chunk_type.startswith('table_row_'),
            'is_table_context': segment.chunk_type.startswith('table_context_')
        }
    }
    documents.append(doc)

# Strategy 1: Insert all chunks for maximum retrieval flexibility
# This gives you full texts + overlapping chunks + table context + table rows

# Strategy 2: Insert only full texts and full tables
full_docs = [d for d in documents if d['metadata']['chunk_type'] == 'full']

# Strategy 3: Insert full texts + table rows + table context (no text chunks)
full_and_tables = [d for d in documents 
                   if d['metadata']['chunk_type'] == 'full' 
                   or d['metadata']['is_table_row']
                   or d['metadata']['is_table_context']]

# Strategy 4: Separate text and table documents for different retrieval strategies
text_docs = [d for d in documents if d['metadata']['content_type'] == 'text']
table_docs = [d for d in documents if d['metadata']['content_type'] == 'table']

# Now insert documents into your vector database
# e.g., Pinecone, Weaviate, ChromaDB, etc.
```

## PDF-Specific Features

1. **Page-level context**: Each segment includes all text and tables from its page
2. **Cross-page context**: Boundary chunks include adjacent pages for better context
3. **Page number tracking**: Easy to reference specific pages in source PDFs
4. **Multi-page document support**: Handles PDFs with multiple pages
5. **Docling compatibility**: Works with JSON output from Docling PDF parser
6. **Table processing with context**: Extracts tables with surrounding text and creates row-level chunks
7. **Mixed content**: Handles pages with both text and tables
8. **Page-level text processing**: Concatenates all text on a page before chunking

## Chunking Benefits for RAG

1. **Better matching**: Smaller chunks can match user queries more precisely
2. **Context preservation**: Full text and entire page data always available
3. **Overlap reduces boundary issues**: Important information at chunk boundaries is captured
4. **Multiple retrieval options**: Can retrieve at different granularities
5. **Improved relevance**: Shorter chunks often have higher semantic coherence
6. **Page-aware retrieval**: Can retrieve all content from a specific page
7. **Table-aware retrieval**: Can match specific table rows while preserving full table context
8. **Table context**: Surrounding text helps understand table meaning
9. **Cross-page awareness**: Boundary chunks include adjacent pages for better context
10. **Natural language table format**: "header is value" format is more readable for LLMs

## Comparison with HTML Processor

| Feature | PDF Processor | HTML Processor |
|---------|--------------|----------------|
| Source | PDF documents | HTML web pages |
| Organization | By page number | By URL |
| Context field | `entire_page_data` | `full_page_text` |
| Text field | `page_chunk` | `text` |
| Additional metadata | `page_number` | `hyperlinks` |
| Table format | "header is value" pairs | N/A |
| Table context | 2 context chunks per table | N/A |
| Cross-page context | Yes (for boundary chunks) | N/A |
| Typical use case | Policy documents, forms | Web content, FAQs |

## API Changes

### Breaking Changes from Previous Version

1. **`get_pdf_segment()` signature changed**:
   - Old: `get_pdf_segment(file_index, content_index)`
   - New: `get_pdf_segment(file_index, page_number)`
   - Now returns all segments for a page instead of a specific content item

2. **Table row format changed**:
   - Old: `header1 | header2\nvalue1 | value2`
   - New: `header1 is value1, header2 is value2`

3. **New chunk types**:
   - `table_context_1`: Text context before table
   - `table_context_2`: Text context + table preview

4. **Cross-page context**:
   - `entire_page_data` may now include adjacent pages for boundary chunks
   - Format: `[PREVIOUS PAGE N]\n...\n\n[CURRENT PAGE N+1]\n...\n\n[NEXT PAGE N+2]\n...`

5. **Page-level processing**:
   - Text is now processed at page level (all text items concatenated first)
   - Chunks are created from full page text, not individual text items
