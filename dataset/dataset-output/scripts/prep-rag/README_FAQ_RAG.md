# FAQ Data Extractor for RAG System

This script prepares FAQ data from `topics-faq.json` for insertion into a RAG (Retrieval-Augmented Generation) system.

## Usage

### Basic Usage

```bash
python3 scripts/prepare_faq_for_rag.py dataset/topics-faq.json
```

This will run a demonstration showing:
- Total FAQ entries and topics
- Sample FAQ entries with all extracted fields
- Topic-specific queries
- Index-based access

### API Usage in Your Code

```python
from prepare_faq_for_rag import FAQDataExtractor

# Initialize the extractor
extractor = FAQDataExtractor('dataset/topics-faq.json')

# Get all FAQ entries
all_entries = extractor.extract_faq_entries()

# Each entry contains:
for entry in all_entries:
    topic = entry['topic']                    # e.g., 'business', 'health', 'car'
    question = entry['question']              # The question text
    formatted_qa = entry['formatted_qa']      # "question: ...\nanswer: ..."
    hyperlinks = entry['hyperlinks']          # List of {'text': xxx, 'link': xxx}
    url = entry['url']                        # Source URL

# Get FAQs for a specific topic
business_faqs = extractor.get_faqs_by_topic('business')

# Get a specific FAQ by index
first_faq = extractor.get_faq_by_index(0)

# Get statistics
total_count = extractor.get_total_count()
topics = extractor.get_topics()
```

## Data Structure

Each FAQ entry returned by the API contains:

| Field | Type | Description |
|-------|------|-------------|
| `topic` | str | Topic category (e.g., 'business', 'health', 'car') |
| `question` | str | The question text |
| `formatted_qa` | str | Question and answer formatted as:<br>`"question: ...\nanswer: ..."` |
| `hyperlinks` | list | List of dicts with `{'text': str, 'link': str}` |
| `url` | str | Source URL for the FAQ |

## Example Output

```python
{
    'topic': 'business',
    'question': 'איך מצטרפים לביטוח עסק של הראל?',
    'formatted_qa': 'question: איך מצטרפים לביטוח עסק של הראל?\nanswer: תהליך ההצטרפות...',
    'hyperlinks': [
        {'text': 'המידע האישי', 'link': 'https://www.harel-group.co.il/...'},
        {'text': 'בקשה לביטול', 'link': 'https://www.harel-group.co.il/...'}
    ],
    'url': 'https://www.harel-group.co.il/insurance/business'
}
```

## Integration with RAG System

The extracted data is ready for RAG system insertion:

1. **Document Text**: Use `formatted_qa` as the document content
2. **Metadata**: Store `topic`, `url`, and `question` as metadata
3. **References**: Store `hyperlinks` for citation and reference tracking
4. **Filtering**: Use `topic` for filtered retrieval

### Example RAG Integration

```python
from prepare_faq_for_rag import FAQDataExtractor

extractor = FAQDataExtractor('dataset/topics-faq.json')
entries = extractor.extract_faq_entries()

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
    # Insert into your RAG system
    # rag_system.add_document(document)
```

## Statistics

The script processes:
- **203 total FAQ entries**
- **8 topics**: dental, health, business, mortgage, life, apartment, travel, car
