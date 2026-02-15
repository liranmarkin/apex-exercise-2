# Policies Data Processor for RAG System

This script processes `topics-policies-manual-html-parse.json` and provides API functions to extract structured data for RAG (Retrieval-Augmented Generation) systems.

## Key Features

- **Filters strHTML fields only**: Only processes fields with `"field": "strHTML"` 
- **URL normalization**: Removes `/index` suffix from URLs
- **Smart chunking**: Automatically creates overlapping chunks for longer texts
- **TextSegment API**: Returns structured data including:
  - Topic
  - Text content (clean HTML)
  - Full page text (all strHTML text from the same URL)
  - Hyperlinks dictionary
  - Source URL
  - Chunk type (full, chunk_1-2, chunk_2-3, etc.)

## Chunking Strategy

When text exceeds a word count threshold (default: 50 words), the processor creates:
1. **Full text**: The complete original text (chunk_type: "full")
2. **2-sentence chunks**: Overlapping pairs (chunk_1-2, chunk_2-3, chunk_3-4, etc.)
3. **3-sentence chunks**: Overlapping triplets (chunk_1-3, chunk_2-4, etc.)
4. **4-sentence chunks**: Overlapping quads (chunk_1-4, chunk_2-5, etc.)

This provides multiple granularities for better RAG retrieval.

## Usage

### Command Line

```bash
# Show statistics (default threshold: 50 words)
python3 scripts/prep-rag/prepare_policies_for_rag.py dataset/topics-policies-manual-html-parse.json

# Show statistics with custom threshold
python3 scripts/prep-rag/prepare_policies_for_rag.py dataset/topics-policies-manual-html-parse.json --chunk-threshold 30

# Run demonstration
python3 scripts/prep-rag/prepare_policies_for_rag.py dataset/topics-policies-manual-html-parse.json --demo

# Run demonstration with custom threshold
python3 scripts/prep-rag/prepare_policies_for_rag.py dataset/topics-policies-manual-html-parse.json --demo --chunk-threshold 30
```

### Python API

```python
from prepare_policies_for_rag import PoliciesDataProcessor

# Initialize processor with custom chunk threshold
processor = PoliciesDataProcessor(
    'dataset/topics-policies-manual-html-parse.json',
    chunk_threshold=50  # Words threshold for chunking
)

# Get specific text segment (returns list of segments if chunked)
segments = processor.get_text_segment(file_index=0, text_index=0)
if segments:
    for segment in segments:
        print(f"Chunk type: {segment.chunk_type}")
        print(f"Topic: {segment.topic}")
        print(f"Text: {segment.text}")
        print(f"URL: {segment.url}")
        print(f"Hyperlinks: {segment.hyperlinks}")
        print(f"Full page: {segment.full_page_text}")

# Iterate through all segments (includes all chunks)
for segment in processor.iter_all_segments():
    if segment.chunk_type == "full":
        print(f"Full text: {segment.text[:50]}...")
    else:
        print(f"Chunk {segment.chunk_type}: {segment.text[:50]}...")

# Get only full segments (no chunks) by topic
travel_segments = processor.get_segments_by_topic('travel')
full_segments = [s for s in travel_segments if s.chunk_type == "full"]
print(f"Found {len(full_segments)} full travel segments")

# Get segments by URL (normalized)
url_segments = processor.get_segments_by_url('https://www.harel-group.co.il/insurance/travel')

# Get statistics
stats = processor.get_statistics()
print(f"Total strHTML segments: {stats['total_strhtml_segments']}")
print(f"Total chunks (including overlaps): {stats['total_chunks_including_overlaps']}")
print(f"Additional chunks created: {stats['additional_chunks_created']}")
print(f"Topics: {stats['topics_list']}")

# Export to JSON
segment_dict = segments[0].to_dict()
import json
json_output = json.dumps(segment_dict, ensure_ascii=False, indent=2)
```

## Data Structure

### TextSegment Object

```python
@dataclass
class TextSegment:
    topic: str                          # Topic category (e.g., 'travel', 'health')
    text: str                           # Clean text content (full or chunk)
    full_page_text: str                 # All strHTML text from the same URL
    hyperlinks: List[Dict[str, str]]    # List of {text, link} dictionaries
    url: str                            # Normalized source URL (no /index suffix)
    chunk_type: str                     # "full", "chunk_1-2", "chunk_2-3", etc.
```

### Example Output

```json
{
  "topic": "travel",
  "text": "ספורט אתגרי חובבני כולל ענפי ספורט...",
  "full_page_text": "ספורט אתגרי חובבני כולל...\n\nאפשר להאריך...",
  "hyperlinks": [
    {
      "text": "לרשימת ענפי הספורט האתגרי",
      "link": "/media/of1evuu4/הודעה-על-הגדרת-ספורט-אתגרי.pdf"
    }
  ],
  "url": "https://www.harel-group.co.il/insurance/travel",
  "chunk_type": "full"
}
```

## Dataset Statistics

With default settings (50-word threshold):
- Total files: 87
- Total strHTML segments: 1,101
- Total chunks (including overlaps): 1,260
- Additional chunks created: 159
- Unique topics: 8 (apartment, business, car, dental, health, life, mortgage, travel)
- Unique URLs: 82 (after normalization)
- Segments with hyperlinks: 40

## Integration with RAG Systems

This processor is designed to prepare data for RAG systems. Each text segment can be:

1. **Embedded**: Use the `text` field for vector embeddings
2. **Contextualized**: Include `full_page_text` for broader context
3. **Enriched**: Use `hyperlinks` for additional references
4. **Categorized**: Filter by `topic` for domain-specific retrieval
5. **Sourced**: Track origin with `url` field
6. **Multi-granular**: Use `chunk_type` to retrieve at different levels

## Example RAG Integration

```python
# Prepare data for vector database
processor = PoliciesDataProcessor('dataset/topics-policies-manual-html-parse.json')

documents = []
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
    documents.append(doc)

# Strategy 1: Insert all chunks for maximum retrieval flexibility
# This gives you full texts + overlapping chunks for better matching

# Strategy 2: Insert only full texts for simpler retrieval
full_docs = [d for d in documents if d['metadata']['chunk_type'] == 'full']

# Strategy 3: Insert full texts + selected chunk sizes
full_and_pairs = [d for d in documents 
                  if d['metadata']['chunk_type'] == 'full' 
                  or d['metadata']['chunk_type'].startswith('chunk_') 
                  and '-2' in d['metadata']['chunk_type']]

# Now insert documents into your vector database
# e.g., Pinecone, Weaviate, ChromaDB, etc.
```

## Chunking Benefits for RAG

1. **Better matching**: Smaller chunks can match user queries more precisely
2. **Context preservation**: Full text is always available
3. **Overlap reduces boundary issues**: Important information at chunk boundaries is captured
4. **Multiple retrieval options**: Can retrieve at different granularities
5. **Improved relevance**: Shorter chunks often have higher semantic coherence
