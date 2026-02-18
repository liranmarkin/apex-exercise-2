#!/usr/bin/env python3
"""
Prepare FAQ JSON data for RAG system insertion.
Extracts topic, question, answer, hyperlinks, and URL for each FAQ entry.
"""

import json
import sys
from typing import List, Dict, Any, Optional


class FAQDataExtractor:
    """Extract and format FAQ data for RAG system."""
    
    def __init__(self, json_file_path: str):
        """Initialize with path to topics-faq.json file."""
        self.json_file_path = json_file_path
        self.data = None
        self._load_data()
    
    def _load_data(self):
        """Load JSON data from file."""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.json_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
    
    def extract_faq_entries(self) -> List[Dict[str, Any]]:
        """
        Extract all FAQ entries from the JSON file.
        
        Returns:
            List of dictionaries, each containing:
            - topic: The topic category
            - question: The question text
            - formatted_qa: Question and answer formatted with prefix
            - hyperlinks: List of hyperlink dictionaries with 'text' and 'link'
            - url: Source URL
        """
        if not self.data or 'content' not in self.data:
            return []
        
        faq_entries = []
        
        for content_item in self.data['content']:
            topic = content_item.get('topic', 'unknown')
            source_url = content_item.get('source_url', '')
            data_section = content_item.get('data', {})
            faqs = data_section.get('faqs', [])
            
            for faq in faqs:
                question = faq.get('question', '')
                answer_text = faq.get('answer_text', '')
                hyperlinks_dict = faq.get('hyperlinks', {})
                
                # Format question and answer with prefix
                formatted_qa = f"שאלה: {question}\nתשובה: {answer_text}"
                
                # Convert hyperlinks dict to list of dicts
                hyperlinks = [
                    {'text': text, 'link': link}
                    for text, link in hyperlinks_dict.items()
                ]
                
                faq_entry = {
                    'topic': topic,
                    'question': question,
                    'answer': answer_text,
                    'formatted_qa': formatted_qa,
                    'hyperlinks': hyperlinks,
                    'url': source_url
                }
                
                faq_entries.append(faq_entry)
        
        return faq_entries
    
    def get_faq_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get a specific FAQ entry by index."""
        entries = self.extract_faq_entries()
        if 0 <= index < len(entries):
            return entries[index]
        return None
    
    def get_faqs_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get all FAQ entries for a specific topic."""
        entries = self.extract_faq_entries()
        return [entry for entry in entries if entry['topic'] == topic]
    
    def get_total_count(self) -> int:
        """Get total number of FAQ entries."""
        return len(self.extract_faq_entries())
    
    def get_topics(self) -> List[str]:
        """Get list of unique topics."""
        entries = self.extract_faq_entries()
        return list(set(entry['topic'] for entry in entries))


def demonstrate_usage(json_file_path: str):
    """Demonstrate how to use the FAQDataExtractor."""
    print("=" * 80)
    print("FAQ Data Extractor - Demonstration")
    print("=" * 80)
    print()
    
    # Initialize extractor
    extractor = FAQDataExtractor(json_file_path)
    
    # Show statistics
    total_count = extractor.get_total_count()
    topics = extractor.get_topics()
    
    print(f"📊 Statistics:")
    print(f"   Total FAQ entries: {total_count}")
    print(f"   Topics: {', '.join(topics)}")
    print()
    
    # Extract all entries
    all_entries = extractor.extract_faq_entries()
    
    # Show first 3 entries as examples
    print("=" * 80)
    print("📝 Sample FAQ Entries (showing first 10)")
    print("=" * 80)
    print()
    
    for i, entry in enumerate(all_entries[:10], 1):
        print(f"Entry #{i}")
        print("-" * 80)
        print(f"Topic: {entry['topic']}")
        print(f"URL: {entry['url']}")
        print()
        print(f"Question: {entry['question']}")
        print()
        print("Formatted Q&A:")
        print(entry['formatted_qa'])
        print()
        
        if entry['hyperlinks']:
            print(f"Hyperlinks ({len(entry['hyperlinks'])}):")
            for link in entry['hyperlinks']:
                print(f"  • {link['text']}")
                print(f"    → {link['link']}")
        else:
            print("Hyperlinks: None")
        
        print()
        print("=" * 80)
        print()
    
    # Show topic-specific query
    if topics:
        sample_topic = topics[0]
        topic_entries = extractor.get_faqs_by_topic(sample_topic)
        print(f"🔍 Topic-specific query: '{sample_topic}'")
        print(f"   Found {len(topic_entries)} entries for this topic")
        print()
    
    # Show how to access by index
    print("🎯 Access by index:")
    entry = extractor.get_faq_by_index(0)
    if entry:
        print(f"   Entry #0 question: {entry['question'][:60]}...")
    print()
    
    print("=" * 80)
    print("✅ Demonstration complete!")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python prepare_faq_for_rag.py <topics-faq.json>")
        print()
        print("Example:")
        print("  python prepare_faq_for_rag.py dataset/topics-faq.json")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    
    try:
        demonstrate_usage(json_file_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
