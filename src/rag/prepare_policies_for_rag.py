#!/usr/bin/env python3
"""
Script to prepare topics-policies-manual-html-parse.json for RAG system.
Provides API functions to extract structured data for each text segment.
"""

import json
import argparse
import re
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass, asdict


@dataclass
class TextSegment:
    """Represents a single text segment with its metadata."""
    topic: str
    text: str
    full_page_text: str
    hyperlinks: List[Dict[str, str]]
    url: str
    chunk_type: str = "full"  # "full", "chunk_1-2", "chunk_2-3", etc.
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PoliciesDataProcessor:
    """Process policies JSON data for RAG system."""
    
    def __init__(self, json_file_path: str, chunk_threshold: int = 50, chunk_overlap: int = 1):
        """
        Initialize processor with JSON file.
        
        Args:
            json_file_path: Path to topics-policies-manual-html-parse.json
            chunk_threshold: Number of words above which to create chunks (default: 50)
            chunk_overlap: Number of sentences to overlap between chunks (default: 1)
        """
        self.json_file_path = json_file_path
        self.chunk_threshold = chunk_threshold
        self.chunk_overlap = chunk_overlap
        self.data = self._load_json()
        self._build_full_page_cache()
    
    def _load_json(self) -> Dict:
        """Load JSON file."""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL by removing /index suffix.
        
        Args:
            url: Original URL
            
        Returns:
            Normalized URL
        """
        if url.endswith('/index'):
            return url[:-6]  # Remove '/index'
        return url
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using basic Hebrew/English sentence detection.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Split on common sentence endings: . ! ? followed by space or end
        # Also handle Hebrew punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]
    
    def _count_words(self, text: str) -> int:
        """
        Count words in text (works for both Hebrew and English).
        
        Args:
            text: Text to count words in
            
        Returns:
            Number of words
        """
        return len(text.split())
    
    def _create_chunks(self, text: str) -> List[tuple]:
        """
        Create overlapping chunks from text if it exceeds threshold.
        
        Returns list of tuples: (chunk_text, chunk_type)
        - First tuple is always the full text with type "full"
        - Additional tuples are chunks with types like "chunk_1-2", "chunk_2-3", etc.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of (chunk_text, chunk_type) tuples
        """
        word_count = self._count_words(text)
        
        # Always include the full text first
        chunks = [(text, "full")]
        
        # Only create additional chunks if text exceeds threshold
        if word_count <= self.chunk_threshold:
            return chunks
        
        sentences = self._split_into_sentences(text)
        
        # If only 1-2 sentences, no need to chunk further
        if len(sentences) <= 2:
            return chunks
        
        # Create overlapping chunks: 1-2, 2-3, 3-4, etc.
        for i in range(len(sentences) - 1):
            chunk_text = ' '.join(sentences[i:i+2])
            chunk_type = f"chunk_{i+1}-{i+2}"
            chunks.append((chunk_text, chunk_type))
        
        # Also create 3-sentence chunks if we have enough sentences
        if len(sentences) >= 3:
            for i in range(len(sentences) - 2):
                chunk_text = ' '.join(sentences[i:i+3])
                chunk_type = f"chunk_{i+1}-{i+3}"
                chunks.append((chunk_text, chunk_type))
        
        # Also create 4-sentence chunks if we have enough sentences
        if len(sentences) >= 4:
            for i in range(len(sentences) - 3):
                chunk_text = ' '.join(sentences[i:i+4])
                chunk_type = f"chunk_{i+1}-{i+4}"
                chunks.append((chunk_text, chunk_type))
        
        return chunks
    
    def _build_full_page_cache(self):
        """Build cache of full page text for each URL (only strHTML fields)."""
        self.full_page_cache = {}
        
        for file_entry in self.data.get('files', []):
            url = self._normalize_url(file_entry.get('url', ''))
            next_data_fields = file_entry.get('next_data_fields', [])
            
            # Concatenate all clean text from strHTML fields only
            full_text_parts = []
            for field in next_data_fields:
                if field.get('field') == 'strHTML':
                    clean_text = field.get('clean', '').strip()
                    if clean_text:
                        full_text_parts.append(clean_text)
            
            self.full_page_cache[url] = '\n\n'.join(full_text_parts)
    
    def get_text_segment(self, file_index: int, text_index: int) -> Optional[List[TextSegment]]:
        """
        Get specific text segment(s) by file and text index.
        Returns multiple segments if text is chunked.
        
        Args:
            file_index: Index of the file in the files array
            text_index: Index of the text segment in next_data_fields array
            
        Returns:
            List of TextSegment objects (multiple if chunked) or None if not found
        """
        try:
            file_entry = self.data['files'][file_index]
            text_field = file_entry['next_data_fields'][text_index]
            
            # Only process strHTML fields
            if text_field.get('field') != 'strHTML':
                return None
            
            url = self._normalize_url(file_entry.get('url', ''))
            topic = file_entry.get('topic', '')
            text = text_field.get('clean', '')
            hyperlinks = text_field.get('hyperlinks', [])
            full_page_text = self.full_page_cache.get(url, '')
            
            # Create chunks
            chunks = self._create_chunks(text)
            
            # Create TextSegment for each chunk
            segments = []
            for chunk_text, chunk_type in chunks:
                segments.append(TextSegment(
                    topic=topic,
                    text=chunk_text,
                    full_page_text=full_page_text,
                    hyperlinks=hyperlinks,
                    url=url,
                    chunk_type=chunk_type
                ))
            
            return segments
        except (IndexError, KeyError):
            return None
    
    def iter_all_segments(self) -> Generator[TextSegment, None, None]:
        """
        Iterator that yields all text segments from all files (only strHTML fields).
        Yields multiple segments per text if chunking is applied.
        
        Yields:
            TextSegment objects
        """
        for file_entry in self.data.get('files', []):
            topic = file_entry.get('topic', '')
            url = self._normalize_url(file_entry.get('url', ''))
            full_page_text = self.full_page_cache.get(url, '')
            
            for text_field in file_entry.get('next_data_fields', []):
                # Only process strHTML fields
                if text_field.get('field') != 'strHTML':
                    continue
                
                text = text_field.get('clean', '')
                hyperlinks = text_field.get('hyperlinks', [])
                
                # Create chunks
                chunks = self._create_chunks(text)
                
                # Yield a segment for each chunk
                for chunk_text, chunk_type in chunks:
                    yield TextSegment(
                        topic=topic,
                        text=chunk_text,
                        full_page_text=full_page_text,
                        hyperlinks=hyperlinks,
                        url=url,
                        chunk_type=chunk_type
                    )
    
    def get_segments_by_topic(self, topic: str) -> List[TextSegment]:
        """
        Get all text segments for a specific topic.
        
        Args:
            topic: Topic name to filter by
            
        Returns:
            List of TextSegment objects
        """
        segments = []
        for segment in self.iter_all_segments():
            if segment.topic == topic:
                segments.append(segment)
        return segments
    
    def get_segments_by_url(self, url: str) -> List[TextSegment]:
        """
        Get all text segments from a specific URL.
        
        Args:
            url: URL to filter by
            
        Returns:
            List of TextSegment objects
        """
        segments = []
        for segment in self.iter_all_segments():
            if segment.url == url:
                segments.append(segment)
        return segments
    
    def get_statistics(self) -> Dict:
        """Get statistics about the dataset."""
        total_segments = 0
        total_chunks = 0
        topics = set()
        urls = set()
        segments_with_links = 0
        chunked_segments = 0
        
        for segment in self.iter_all_segments():
            total_chunks += 1
            if segment.chunk_type == "full":
                total_segments += 1
                if segment.hyperlinks:
                    segments_with_links += 1
            else:
                chunked_segments += 1
            topics.add(segment.topic)
            urls.add(segment.url)
        
        return {
            'total_files': self.data.get('total_files', 0),
            'total_strhtml_segments': total_segments,
            'total_chunks_including_overlaps': total_chunks,
            'additional_chunks_created': chunked_segments,
            'unique_topics': len(topics),
            'unique_urls': len(urls),
            'segments_with_hyperlinks': segments_with_links,
            'topics_list': sorted(list(topics)),
            'chunk_threshold_words': self.chunk_threshold
        }


def demonstrate_usage(json_file_path: str, chunk_threshold: int = 50):
    """
    Demonstration function showing how to use the API.
    
    Args:
        json_file_path: Path to the JSON file
        chunk_threshold: Word count threshold for chunking
    """
    print("=" * 80)
    print("POLICIES DATA PROCESSOR - DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize processor
    processor = PoliciesDataProcessor(json_file_path, chunk_threshold=chunk_threshold)
    
    # Show statistics
    print("📊 DATASET STATISTICS")
    print("-" * 80)
    stats = processor.get_statistics()
    for key, value in stats.items():
        if key != 'topics_list':
            print(f"  {key}: {value}")
    print(f"  Available topics: {', '.join(stats['topics_list'])}")
    print()
    
    # Example 1: Get first text segment (may return multiple if chunked)
    print("📄 EXAMPLE 1: Get specific text segment (file_index=0, text_index=0)")
    print("-" * 80)
    segments = processor.get_text_segment(0, 0)
    if segments:
        for i, segment in enumerate(segments):
            print(f"  Chunk {i+1}/{len(segments)} (type: {segment.chunk_type}):")
            print(f"    Topic: {segment.topic}")
            print(f"    URL: {segment.url}")
            print(f"    Text: {segment.text[:100]}..." if len(segment.text) > 100 else f"    Text: {segment.text}")
            print(f"    Word count: {len(segment.text.split())} words")
            if i == 0:  # Only show these for first chunk
                print(f"    Hyperlinks: {len(segment.hyperlinks)} link(s)")
                if segment.hyperlinks:
                    for link in segment.hyperlinks[:2]:
                        print(f"      - {link.get('text', '')}: {link.get('link', '')}")
                print(f"    Full page text length: {len(segment.full_page_text)} characters")
            print()
    
    # Example 2: Show chunking in action
    print("📋 EXAMPLE 2: Demonstrate chunking on longer text")
    print("-" * 80)
    found_long_text = False
    for segment in processor.iter_all_segments():
        if segment.chunk_type == "full" and len(segment.text.split()) > chunk_threshold:
            print(f"  Found text with {len(segment.text.split())} words (threshold: {chunk_threshold})")
            print(f"  URL: {segment.url}")
            print(f"  Full text: {segment.text[:150]}...")
            print()
            
            # Get all chunks for this segment
            # We need to find it again to get all chunks
            for file_idx, file_entry in enumerate(processor.data.get('files', [])):
                if processor._normalize_url(file_entry.get('url', '')) == segment.url:
                    for text_idx, text_field in enumerate(file_entry.get('next_data_fields', [])):
                        if text_field.get('field') == 'strHTML' and text_field.get('clean', '') == segment.text:
                            all_chunks = processor.get_text_segment(file_idx, text_idx)
                            if all_chunks and len(all_chunks) > 1:
                                print(f"  Created {len(all_chunks)} chunks:")
                                for chunk in all_chunks:
                                    print(f"    - {chunk.chunk_type}: {chunk.text[:80]}...")
                            found_long_text = True
                            break
                if found_long_text:
                    break
            break
    
    if not found_long_text:
        print(f"  No text found exceeding {chunk_threshold} words threshold")
    print()
    
    # Example 3: Get segments by topic
    if stats['topics_list']:
        first_topic = stats['topics_list'][0]
        print(f"🏷️  EXAMPLE 3: Get segments for topic '{first_topic}'")
        print("-" * 80)
        topic_segments = processor.get_segments_by_topic(first_topic)
        full_segments = [s for s in topic_segments if s.chunk_type == "full"]
        print(f"  Found {len(full_segments)} full segments for topic '{first_topic}'")
        print(f"  Total including chunks: {len(topic_segments)} segments")
        if topic_segments:
            print(f"  First segment text: {topic_segments[0].text[:100]}...")
        print()
    
    # Example 4: Export to JSON format
    print("💾 EXAMPLE 4: Export segment to JSON format")
    print("-" * 80)
    segments = processor.get_text_segment(0, 0)
    if segments:
        segment = segments[0]  # Show first chunk
        json_output = json.dumps(segment.to_dict(), ensure_ascii=False, indent=2)
        print(f"  JSON output (truncated):")
        lines = json_output.split('\n')[:20]
        for line in lines:
            print(f"  {line}")
        if len(json_output.split('\n')) > 20:
            print("  ...")
    print()
    
    print("=" * 80)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Prepare policies JSON data for RAG system'
    )
    parser.add_argument(
        'json_file',
        help='Path to topics-policies-manual-html-parse.json'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demonstration of API usage'
    )
    parser.add_argument(
        '--chunk-threshold',
        type=int,
        default=50,
        help='Number of words above which to create chunks (default: 50)'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        demonstrate_usage(args.json_file, chunk_threshold=args.chunk_threshold)
    else:
        # If not demo mode, just show basic info
        processor = PoliciesDataProcessor(args.json_file, chunk_threshold=args.chunk_threshold)
        stats = processor.get_statistics()
        print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
