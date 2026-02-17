#!/usr/bin/env python3
"""
Script to prepare topics-pdf-docling.json for RAG system.
Provides API functions to extract structured data for each PDF page chunk.
"""

import json
import argparse
import re
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass, asdict


@dataclass
class PDFSegment:
    """Represents a single PDF page chunk with its metadata."""
    topic: str
    page_chunk: str
    entire_page_data: str
    page_number: int
    url: str
    chunk_type: str = "full"  # "full", "chunk_1-2", "chunk_2-3", etc.
    content_type: str = "text"  # "text" or "table"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PDFDataProcessor:
    """Process PDF docling JSON data for RAG system."""
    
    def __init__(self, json_file_path: str, chunk_threshold: int = 50):
        """
        Initialize processor with JSON file.
        
        Args:
            json_file_path: Path to topics-pdf-docling.json
            chunk_threshold: Number of words above which to create chunks (default: 50)
        """
        self.json_file_path = json_file_path
        self.chunk_threshold = chunk_threshold
        self.data = self._load_json()
        self._build_page_cache()
    
    def _load_json(self) -> Dict:
        """Load JSON file."""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter out files with 'ערבית' in the filename
        if 'files' in data:
            original_count = len(data['files'])
            data['files'] = [
                f for f in data['files'] 
                if 'ערבית' not in f.get('file_path', '') and 'ערבית' not in f.get('url', '')
            ]
            filtered_count = original_count - len(data['files'])
            if filtered_count > 0:
                print(f"ℹ️  Filtered out {filtered_count} Arabic PDF(s) (containing 'ערבית')")
        
        return data
    
    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL by removing /index suffix.
        
        Args:
            url: Original URL
            
        Returns:
            Normalized URL
        """
        if url.endswith('/index'):
            return url[:-6]
        return url
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by cleaning unicode artifacts and extra whitespace.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return text
        
        # Remove unicode artifacts like /uniXXXX (e.g., /uniFEEE)
        text = re.sub(r'/uni[0-9A-Fa-f]{4}', '', text)
        
        # Replace various unicode spaces with regular space
        # Including: non-breaking space, thin space, hair space, etc.
        unicode_spaces = [
            '\u00A0',  # Non-breaking space
            '\u2000', '\u2001', '\u2002', '\u2003', '\u2004',  # Various em/en spaces
            '\u2005', '\u2006', '\u2007', '\u2008', '\u2009',  # Thin/hair spaces
            '\u200A', '\u200B',  # Hair space, zero-width space
            '\u202F',  # Narrow no-break space
            '\u205F',  # Medium mathematical space
            '\u3000',  # Ideographic space
            '\uFEFF',  # Zero-width no-break space (BOM)
        ]
        for unicode_space in unicode_spaces:
            text = text.replace(unicode_space, ' ')
        
        # Collapse multiple whitespace (spaces, tabs, newlines within text) into single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Clean up but preserve intentional line breaks
        # Remove spaces at start/end of lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        # Remove multiple consecutive newlines (keep max 2 for paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using basic Hebrew/English sentence detection.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _count_words(self, text: str) -> int:
        """
        Count words in text.
        
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
        
        if len(sentences) <= 2:
            return chunks
        
        # Create overlapping chunks
        for i in range(len(sentences) - 1):
            chunk_text = ' '.join(sentences[i:i+2])
            chunk_type = f"chunk_{i+1}-{i+2}"
            chunks.append((chunk_text, chunk_type))
        
        if len(sentences) >= 3:
            for i in range(len(sentences) - 2):
                chunk_text = ' '.join(sentences[i:i+3])
                chunk_type = f"chunk_{i+1}-{i+3}"
                chunks.append((chunk_text, chunk_type))
        
        if len(sentences) >= 4:
            for i in range(len(sentences) - 3):
                chunk_text = ' '.join(sentences[i:i+4])
                chunk_type = f"chunk_{i+1}-{i+4}"
                chunks.append((chunk_text, chunk_type))
        
        return chunks
    
    def _build_page_cache(self):
        """Build cache of full page text for each URL and page number."""
        self.page_cache = {}
        self.table_cache = {}  # Cache for full table data
        
        for file_entry in self.data.get('files', []):
            url = self._normalize_url(file_entry.get('url', ''))
            content = file_entry.get('content', [])
            
            # Group content by page number
            pages = {}
            for item in content:
                page_num = item.get('page_number')
                if page_num is None:
                    continue
                    
                if item.get('type') == 'text':
                    text = item.get('text', '').strip()
                    # Normalize text
                    text = self._normalize_text(text)
                    if page_num not in pages:
                        pages[page_num] = []
                    if text:
                        pages[page_num].append(text)
                        
                elif item.get('type') == 'table':
                    # Store table representation
                    table_text = self._table_to_text(item.get('table', {}))
                    # Normalize table text
                    table_text = self._normalize_text(table_text)
                    if page_num not in pages:
                        pages[page_num] = []
                    if table_text:
                        pages[page_num].append(f"[TABLE]\n{table_text}")
            
            # Store concatenated text for each page
            for page_num, texts in pages.items():
                cache_key = f"{url}#page{page_num}"
                self.page_cache[cache_key] = '\n'.join(texts)
    
    def _table_to_text(self, table: Dict) -> str:
        """
        Convert table structure to text representation.
        
        Args:
            table: Table dictionary with rows
            
        Returns:
            Text representation of the table
        """
        if not table or 'rows' not in table:
            return ""
        
        rows = table.get('rows', [])
        if not rows:
            return ""
        
        # Build table as text
        lines = []
        for row in rows:
            cells = row.get('cells', [])
            # Sort cells by column index
            sorted_cells = sorted(cells, key=lambda c: c.get('col_index', 0))
            # Normalize cell text
            cell_texts = [self._normalize_text(cell.get('text', '')) for cell in sorted_cells]
            row_text = ' | '.join(cell_texts)
            lines.append(row_text)
        
        return '\n'.join(lines)
    
    def _extract_table_headers(self, table: Dict) -> List[str]:
        """
        Extract header row from table.
        
        Args:
            table: Table dictionary with rows
            
        Returns:
            List of header texts
        """
        if not table or 'rows' not in table:
            return []
        
        rows = table.get('rows', [])
        if not rows:
            return []
        
        # Find header row (usually first row with is_header=true cells)
        for row in rows:
            cells = row.get('cells', [])
            if any(cell.get('is_header', False) for cell in cells):
                sorted_cells = sorted(cells, key=lambda c: c.get('col_index', 0))
                return [self._normalize_text(cell.get('text', '')) for cell in sorted_cells]
        
        # If no explicit headers, use first row
        first_row = rows[0]
        cells = first_row.get('cells', [])
        sorted_cells = sorted(cells, key=lambda c: c.get('col_index', 0))
        return [self._normalize_text(cell.get('text', '')) for cell in sorted_cells]
    
    def _create_table_chunks(self, table: Dict) -> List[tuple]:
        """
        Create chunks from table: each chunk is headers + one data row.
        
        Args:
            table: Table dictionary with rows
            
        Returns:
            List of (chunk_text, chunk_type) tuples
        """
        if not table or 'rows' not in table:
            return []
        
        rows = table.get('rows', [])
        if not rows:
            return []
        
        # Get full table text
        full_table = self._table_to_text(table)
        chunks = [(full_table, "full")]
        
        # Extract headers
        headers = self._extract_table_headers(table)
        header_text = ' | '.join(headers)
        
        # Find first data row index (skip header rows)
        data_start_idx = 0
        for i, row in enumerate(rows):
            cells = row.get('cells', [])
            if not any(cell.get('is_header', False) for cell in cells):
                data_start_idx = i
                break
        
        # Create chunk for each data row (headers + row)
        for i in range(data_start_idx, len(rows)):
            row = rows[i]
            cells = row.get('cells', [])
            sorted_cells = sorted(cells, key=lambda c: c.get('col_index', 0))
            # Normalize cell text
            cell_texts = [self._normalize_text(cell.get('text', '')) for cell in sorted_cells]
            row_text = ' | '.join(cell_texts)
            
            chunk_text = f"{header_text}\n{row_text}"
            chunk_type = f"table_row_{i - data_start_idx + 1}"
            chunks.append((chunk_text, chunk_type))
        
        return chunks
    
    def get_pdf_segment(self, file_index: int, content_index: int) -> Optional[List[PDFSegment]]:
        """
        Get specific PDF segment(s) by file and content index.
        Returns multiple segments if text is chunked or if it's a table with rows.
        
        Args:
            file_index: Index of the file in the files array
            content_index: Index of the content item in content array
            
        Returns:
            List of PDFSegment objects (multiple if chunked/table) or None if not found
        """
        try:
            file_entry = self.data['files'][file_index]
            content_item = file_entry['content'][content_index]
            
            url = self._normalize_url(file_entry.get('url', ''))
            topic = file_entry.get('topic', '')
            page_number = content_item.get('page_number')
            
            if page_number is None:
                return None
            
            # Get entire page data
            cache_key = f"{url}#page{page_number}"
            entire_page_data = self.page_cache.get(cache_key, '')
            
            content_type = content_item.get('type')
            
            if content_type == 'text':
                # Process text content
                text = content_item.get('text', '').strip()
                # Normalize text
                text = self._normalize_text(text)
                if not text:
                    return None
                
                # Create chunks
                chunks = self._create_chunks(text)
                
                # Create PDFSegment for each chunk
                segments = []
                for chunk_text, chunk_type in chunks:
                    segments.append(PDFSegment(
                        topic=topic,
                        page_chunk=chunk_text,
                        entire_page_data=entire_page_data,
                        page_number=page_number,
                        url=url,
                        chunk_type=chunk_type,
                        content_type='text'
                    ))
                
                return segments
                
            elif content_type == 'table':
                # Process table content
                table = content_item.get('table', {})
                if not table:
                    return None
                
                # Create table chunks (full table + header+row chunks)
                chunks = self._create_table_chunks(table)
                
                # Create PDFSegment for each chunk
                segments = []
                for chunk_text, chunk_type in chunks:
                    segments.append(PDFSegment(
                        topic=topic,
                        page_chunk=chunk_text,
                        entire_page_data=entire_page_data,
                        page_number=page_number,
                        url=url,
                        chunk_type=chunk_type,
                        content_type='table'
                    ))
                
                return segments
            
            return None
            
        except (IndexError, KeyError):
            return None
    
    def iter_all_segments(self) -> Generator[PDFSegment, None, None]:
        """
        Iterator that yields all PDF segments from all files.
        Yields multiple segments per text if chunking is applied.
        Yields multiple segments per table (full table + header+row chunks).
        
        Yields:
            PDFSegment objects
        """
        for file_entry in self.data.get('files', []):
            topic = file_entry.get('topic', '')
            url = self._normalize_url(file_entry.get('url', ''))
            
            for content_item in file_entry.get('content', []):
                content_type = content_item.get('type')
                page_number = content_item.get('page_number')
                
                if page_number is None:
                    continue
                
                # Get entire page data
                cache_key = f"{url}#page{page_number}"
                entire_page_data = self.page_cache.get(cache_key, '')
                
                if content_type == 'text':
                    # Process text content
                    text = content_item.get('text', '').strip()
                    # Normalize text
                    text = self._normalize_text(text)
                    if not text:
                        continue
                    
                    # Create chunks
                    chunks = self._create_chunks(text)
                    
                    # Yield a segment for each chunk
                    for chunk_text, chunk_type in chunks:
                        yield PDFSegment(
                            topic=topic,
                            page_chunk=chunk_text,
                            entire_page_data=entire_page_data,
                            page_number=page_number,
                            url=url,
                            chunk_type=chunk_type,
                            content_type='text'
                        )
                        
                elif content_type == 'table':
                    # Process table content
                    table = content_item.get('table', {})
                    if not table:
                        continue
                    
                    # Create table chunks
                    chunks = self._create_table_chunks(table)
                    
                    # Yield a segment for each chunk
                    for chunk_text, chunk_type in chunks:
                        yield PDFSegment(
                            topic=topic,
                            page_chunk=chunk_text,
                            entire_page_data=entire_page_data,
                            page_number=page_number,
                            url=url,
                            chunk_type=chunk_type,
                            content_type='table'
                        )
    
    def get_segments_by_topic(self, topic: str) -> List[PDFSegment]:
        """
        Get all PDF segments for a specific topic.
        
        Args:
            topic: Topic name to filter by
            
        Returns:
            List of PDFSegment objects
        """
        segments = []
        for segment in self.iter_all_segments():
            if segment.topic == topic:
                segments.append(segment)
        return segments
    
    def get_segments_by_url(self, url: str) -> List[PDFSegment]:
        """
        Get all PDF segments from a specific URL.
        
        Args:
            url: URL to filter by
            
        Returns:
            List of PDFSegment objects
        """
        segments = []
        for segment in self.iter_all_segments():
            if segment.url == url:
                segments.append(segment)
        return segments
    
    def get_segments_by_page(self, url: str, page_number: int) -> List[PDFSegment]:
        """
        Get all PDF segments from a specific page.
        
        Args:
            url: URL to filter by
            page_number: Page number to filter by
            
        Returns:
            List of PDFSegment objects
        """
        segments = []
        for segment in self.iter_all_segments():
            if segment.url == url and segment.page_number == page_number:
                segments.append(segment)
        return segments
    
    def get_statistics(self) -> Dict:
        """Get statistics about the dataset."""
        total_segments = 0
        total_chunks = 0
        topics = set()
        urls = set()
        pages = set()
        chunked_segments = 0
        text_segments = 0
        table_segments = 0
        table_rows = 0
        
        for segment in self.iter_all_segments():
            total_chunks += 1
            if segment.chunk_type == "full":
                total_segments += 1
                if segment.content_type == 'text':
                    text_segments += 1
                elif segment.content_type == 'table':
                    table_segments += 1
            else:
                chunked_segments += 1
                if segment.content_type == 'table' and segment.chunk_type.startswith('table_row_'):
                    table_rows += 1
            topics.add(segment.topic)
            urls.add(segment.url)
            pages.add(f"{segment.url}#page{segment.page_number}")
        
        return {
            'total_files': self.data.get('total_files', 0),
            'total_segments': total_segments,
            'text_segments': text_segments,
            'table_segments': table_segments,
            'table_row_chunks': table_rows,
            'total_chunks_including_overlaps': total_chunks,
            'additional_chunks_created': chunked_segments,
            'unique_topics': len(topics),
            'unique_urls': len(urls),
            'unique_pages': len(pages),
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
    print("PDF DATA PROCESSOR - DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize processor
    processor = PDFDataProcessor(json_file_path, chunk_threshold=chunk_threshold)
    
    # Show statistics
    print("📊 DATASET STATISTICS")
    print("-" * 80)
    stats = processor.get_statistics()
    for key, value in stats.items():
        if key != 'topics_list':
            print(f"  {key}: {value}")
    print(f"  Available topics: {', '.join(stats['topics_list'])}")
    print()
    
    # Example 1: Get first PDF segment
    print("📄 EXAMPLE 1: Get specific PDF segment (file_index=0, content_index=0)")
    print("-" * 80)
    segments = processor.get_pdf_segment(0, 0)
    if segments:
        for i, segment in enumerate(segments):
            print(f"  Chunk {i+1}/{len(segments)} (type: {segment.chunk_type}):")
            print(f"    Topic: {segment.topic}")
            print(f"    URL: {segment.url}")
            print(f"    Page: {segment.page_number}")
            print(f"    Text: {segment.page_chunk[:100]}..." if len(segment.page_chunk) > 100 else f"    Text: {segment.page_chunk}")
            print(f"    Word count: {len(segment.page_chunk.split())} words")
            if i == 0:
                print(f"    Entire page length: {len(segment.entire_page_data)} characters")
            print()
    
    # Example 2: Show chunking in action
    print("📋 EXAMPLE 2: Demonstrate chunking on longer text")
    print("-" * 80)
    found_long_text = False
    for segment in processor.iter_all_segments():
        if segment.chunk_type == "full" and len(segment.page_chunk.split()) > chunk_threshold:
            print(f"  Found text with {len(segment.page_chunk.split())} words (threshold: {chunk_threshold})")
            print(f"  URL: {segment.url}")
            print(f"  Page: {segment.page_number}")
            print(f"  Full text: {segment.page_chunk[:150]}...")
            print()
            
            # Get all chunks for this page
            page_segments = processor.get_segments_by_page(segment.url, segment.page_number)
            # Filter to just this specific text's chunks
            matching_chunks = [s for s in page_segments if s.page_chunk == segment.page_chunk or segment.page_chunk in s.page_chunk]
            
            if len(matching_chunks) > 1:
                print(f"  Created {len(matching_chunks)} chunks:")
                for chunk in matching_chunks[:5]:  # Show first 5
                    print(f"    - {chunk.chunk_type}: {chunk.page_chunk[:80]}...")
            
            found_long_text = True
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
            print(f"  First segment text: {topic_segments[0].page_chunk[:100]}...")
            print(f"  First segment page: {topic_segments[0].page_number}")
        print()
    
    # Example 4: Show table processing
    print("📊 EXAMPLE 4: Demonstrate table processing")
    print("-" * 80)
    found_table = False
    for segment in processor.iter_all_segments():
        if segment.content_type == 'table' and segment.chunk_type == "full":
            print(f"  Found table on page {segment.page_number}")
            print(f"  URL: {segment.url}")
            print(f"  Full table:")
            print(f"    {segment.page_chunk[:200]}...")
            print()
            
            # Get all chunks for this table
            page_segments = processor.get_segments_by_page(segment.url, segment.page_number)
            table_chunks = [s for s in page_segments 
                          if s.content_type == 'table' and s.page_chunk == segment.page_chunk]
            
            # Find row chunks
            row_chunks = [s for s in page_segments 
                         if s.content_type == 'table' and s.chunk_type.startswith('table_row_')]
            
            if row_chunks:
                print(f"  Created {len(row_chunks)} row chunks (headers + individual rows):")
                for i, chunk in enumerate(row_chunks[:3]):  # Show first 3
                    print(f"    - {chunk.chunk_type}:")
                    lines = chunk.page_chunk.split('\n')
                    for line in lines[:2]:  # Show header and row
                        print(f"      {line[:70]}...")
                    print()
            
            found_table = True
            break
    
    if not found_table:
        print("  No tables found in dataset")
    print()
    
    # Example 5: Get segments by page
    print("📖 EXAMPLE 5: Get all segments from a specific page")
    print("-" * 80)
    # Find a page with multiple segments
    for segment in processor.iter_all_segments():
        if segment.chunk_type == "full":
            page_segments = processor.get_segments_by_page(segment.url, segment.page_number)
            full_page_segments = [s for s in page_segments if s.chunk_type == "full"]
            if len(full_page_segments) > 1:
                print(f"  URL: {segment.url}")
                print(f"  Page: {segment.page_number}")
                print(f"  Found {len(full_page_segments)} segments on this page")
                print(f"  Total including chunks: {len(page_segments)} segments")
                for i, seg in enumerate(full_page_segments[:3]):
                    content_label = "TABLE" if seg.content_type == 'table' else "TEXT"
                    print(f"    {i+1}. [{content_label}] {seg.page_chunk[:60]}...")
                break
    print()
    
    # Example 6: Export to JSON format
    print("💾 EXAMPLE 6: Export segment to JSON format")
    print("-" * 80)
    segments = processor.get_pdf_segment(0, 0)
    if segments:
        segment = segments[0]
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
        description='Prepare PDF docling JSON data for RAG system'
    )
    parser.add_argument(
        'json_file',
        help='Path to topics-pdf-docling.json'
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
        processor = PDFDataProcessor(args.json_file, chunk_threshold=args.chunk_threshold)
        stats = processor.get_statistics()
        print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
