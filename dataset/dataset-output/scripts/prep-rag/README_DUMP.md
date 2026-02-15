# Dataset Dump Script

This script provides a unified interface to dump all datasets for RAG system insertion. It uses the APIs from all three processors (policies, PDF, FAQ) and outputs structured JSON files.

## Features

- **4 dump functions**: One for each dataset + one combined function
- **Unified CLI**: Single command to dump all or individual datasets
- **Structured output**: JSON files with statistics and segments
- **Configurable chunking**: Adjust word threshold for text chunking
- **Summary generation**: Combined statistics across all sources

## Usage

### Dump All Datasets

```bash
python3 scripts/prep-rag/dump_all_datasets.py --all \
  --policies dataset/topics-policies-manual-html-parse.json \
  --pdf dataset/topics-pdf-docling.json \
  --faq dataset/topics-faq.json \
  --output-dir rag_dumps
```

### Dump Individual Datasets

**Policies (HTML)**:
```bash
python3 scripts/prep-rag/dump_all_datasets.py \
  --policies dataset/topics-policies-manual-html-parse.json \
  --output-dir rag_dumps
```

**PDF (Docling)**:
```bash
python3 scripts/prep-rag/dump_all_datasets.py \
  --pdf dataset/topics-pdf-docling.json \
  --output-dir rag_dumps
```

**FAQ**:
```bash
python3 scripts/prep-rag/dump_all_datasets.py \
  --faq dataset/topics-faq.json \
  --output-dir rag_dumps
```

### Custom Chunk Threshold

```bash
python3 scripts/prep-rag/dump_all_datasets.py --all \
  --policies dataset/topics-policies-manual-html-parse.json \
  --pdf dataset/topics-pdf-docling.json \
  --faq dataset/topics-faq.json \
  --output-dir rag_dumps \
  --chunk-threshold 30
```

## Output Files

When dumping all datasets, the following files are created:

```
rag_dumps/
├── policies_dump.json      # All policies segments (~6 MB)
├── pdf_dump.json           # All PDF segments (~219 MB)
├── faq_dump.json           # All FAQ entries (~222 KB)
└── dump_summary.json       # Combined statistics
```

## Output Format

### Policies Dump (`policies_dump.json`)

```json
{
  "source": "policies-html",
  "statistics": {
    "total_files": 87,
    "total_strhtml_segments": 1101,
    "total_chunks_including_overlaps": 1260,
    "topics_list": ["apartment", "business", "car", ...]
  },
  "segments": [
    {
      "topic": "travel",
      "text": "ספורט אתגרי חובבני...",
      "full_page_text": "...",
      "hyperlinks": [...],
      "url": "https://...",
      "chunk_type": "full"
    },
    ...
  ]
}
```

### PDF Dump (`pdf_dump.json`)

```json
{
  "source": "pdf-docling",
  "statistics": {
    "total_files": 178,
    "total_segments": 42156,
    "text_segments": 41904,
    "table_segments": 252,
    "table_row_chunks": 1964,
    "total_chunks_including_overlaps": 49388,
    "topics_list": ["apartment", "business", "car", ...]
  },
  "segments": [
    {
      "topic": "travel",
      "page_chunk": "ספורט אתגרי...",
      "entire_page_data": "...",
      "page_number": 1,
      "url": "https://...",
      "chunk_type": "full",
      "content_type": "text"
    },
    {
      "topic": "travel",
      "page_chunk": "השתתפות עצמית | האחריות גבול | הכיסוי\nה.ע ללא | $5,000,000 | רפואיות הוצאות",
      "entire_page_data": "...",
      "page_number": 1,
      "url": "https://...",
      "chunk_type": "table_row_1",
      "content_type": "table"
    },
    ...
  ]
}
```

### FAQ Dump (`faq_dump.json`)

```json
{
  "source": "faq",
  "statistics": {
    "total_entries": 203,
    "topics": ["travel", "business", "health", ...]
  },
  "entries": [
    {
      "topic": "travel",
      "question": "מה נחשב כספורט אתגרי?",
      "formatted_qa": "שאלה: מה נחשב כספורט אתגרי?\nתשובה: ...",
      "hyperlinks": [
        {"text": "לרשימה", "link": "https://..."}
      ],
      "url": "https://..."
    },
    ...
  ]
}
```

### Summary (`dump_summary.json`)

```json
{
  "total_segments": 50851,
  "policies_segments": 1260,
  "pdf_segments": 49388,
  "faq_entries": 203,
  "output_files": {
    "policies": "rag_dumps/policies_dump.json",
    "pdf": "rag_dumps/pdf_dump.json",
    "faq": "rag_dumps/faq_dump.json"
  }
}
```

## Python API

You can also use the dump functions programmatically:

```python
from dump_all_datasets import (
    dump_policies_data,
    dump_pdf_data,
    dump_faq_data,
    dump_all_data
)

# Dump individual datasets
policies = dump_policies_data(
    'dataset/topics-policies-manual-html-parse.json',
    output_path='output/policies.json',
    chunk_threshold=50
)

pdf = dump_pdf_data(
    'dataset/topics-pdf-docling.json',
    output_path='output/pdf.json',
    chunk_threshold=50
)

faq = dump_faq_data(
    'dataset/topics-faq.json',
    output_path='output/faq.json'
)

# Or dump all at once
all_data = dump_all_data(
    policies_path='dataset/topics-policies-manual-html-parse.json',
    pdf_path='dataset/topics-pdf-docling.json',
    faq_path='dataset/topics-faq.json',
    output_dir='output',
    chunk_threshold=50
)

# Access the data
print(f"Total policies segments: {len(all_data['policies'])}")
print(f"Total PDF segments: {len(all_data['pdf'])}")
print(f"Total FAQ entries: {len(all_data['faq'])}")
```

## Statistics

With default settings (50-word threshold):

| Dataset | Files | Segments | Size |
|---------|-------|----------|------|
| Policies (HTML) | 87 | 1,260 | ~6 MB |
| PDF (Docling) | 178 | 49,388 | ~219 MB |
| FAQ | - | 203 | ~222 KB |
| **Total** | **265** | **50,851** | **~225 MB** |

## Use Cases

### 1. RAG System Insertion

```python
from dump_all_datasets import dump_all_data
import chromadb

# Dump all data
data = dump_all_data(
    policies_path='dataset/topics-policies-manual-html-parse.json',
    pdf_path='dataset/topics-pdf-docling.json',
    faq_path='dataset/topics-faq.json',
    output_dir='rag_dumps'
)

# Insert into ChromaDB
client = chromadb.Client()
collection = client.create_collection("harel_insurance")

# Insert policies
for segment in data['policies']:
    collection.add(
        documents=[segment['text']],
        metadatas=[{
            'source': 'policies',
            'topic': segment['topic'],
            'url': segment['url'],
            'chunk_type': segment['chunk_type']
        }],
        ids=[f"policy_{hash(segment['text'])}"]
    )

# Insert PDF segments
for segment in data['pdf']:
    collection.add(
        documents=[segment['page_chunk']],
        metadatas=[{
            'source': 'pdf',
            'topic': segment['topic'],
            'url': segment['url'],
            'page': segment['page_number'],
            'content_type': segment['content_type'],
            'chunk_type': segment['chunk_type']
        }],
        ids=[f"pdf_{hash(segment['page_chunk'])}"]
    )

# Insert FAQ entries
for entry in data['faq']:
    collection.add(
        documents=[entry['formatted_qa']],
        metadatas=[{
            'source': 'faq',
            'topic': entry['topic'],
            'url': entry['url'],
            'question': entry['question']
        }],
        ids=[f"faq_{hash(entry['question'])}"]
    )
```

### 2. Data Analysis

```python
from dump_all_datasets import dump_all_data

# Dump and analyze
data = dump_all_data(
    policies_path='dataset/topics-policies-manual-html-parse.json',
    pdf_path='dataset/topics-pdf-docling.json',
    faq_path='dataset/topics-faq.json',
    output_dir='rag_dumps'
)

# Analyze by topic
from collections import Counter

policy_topics = Counter(s['topic'] for s in data['policies'])
pdf_topics = Counter(s['topic'] for s in data['pdf'])
faq_topics = Counter(e['topic'] for e in data['faq'])

print("Policies by topic:", policy_topics)
print("PDF segments by topic:", pdf_topics)
print("FAQ entries by topic:", faq_topics)

# Analyze content types in PDF
pdf_content_types = Counter(s['content_type'] for s in data['pdf'])
print("PDF content types:", pdf_content_types)

# Analyze chunk types
pdf_chunk_types = Counter(s['chunk_type'] for s in data['pdf'])
print("PDF chunk types:", pdf_chunk_types)
```

### 3. Export for External Tools

```bash
# Dump all datasets
python3 scripts/prep-rag/dump_all_datasets.py --all \
  --policies dataset/topics-policies-manual-html-parse.json \
  --pdf dataset/topics-pdf-docling.json \
  --faq dataset/topics-faq.json \
  --output-dir exports

# Now you can:
# - Upload to cloud storage
# - Import into vector databases
# - Process with other tools
# - Share with team members
```

## Performance

- **Policies**: ~1 second to process
- **PDF**: ~10-15 seconds to process (large dataset)
- **FAQ**: <1 second to process
- **Total**: ~15-20 seconds for all datasets

## Requirements

- Python 3.7+
- All three processor modules must be in the same directory:
  - `prepare_policies_for_rag.py`
  - `prepare_pdf_for_rag.py`
  - `prepare_faq_for_rag.py`

## Troubleshooting

**Import Error**:
```
ModuleNotFoundError: No module named 'prepare_policies_for_rag'
```
Solution: Make sure all processor files are in the same directory as `dump_all_datasets.py`

**Memory Error** (for large datasets):
```
MemoryError: Unable to allocate array
```
Solution: Process datasets individually or increase available memory

**File Not Found**:
```
FileNotFoundError: [Errno 2] No such file or directory
```
Solution: Check that dataset paths are correct and files exist
