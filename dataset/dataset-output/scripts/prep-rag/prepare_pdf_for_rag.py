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
    
    def _get_cross_page_context(self, url: str, page_number: int, chunk_text: str) -> str:
        """
        Get context that includes adjacent pages if chunk might span pages.
        
        Args:
            url: URL of the document
            page_number: Current page number
            chunk_text: The chunk text to check
            
        Returns:
            Context string that may include adjacent pages
        """
        cache_key = f"{url}#page{page_number}"
        current_page = self.page_cache.get(cache_key, '')
        
        # Check if we need previous page context
        prev_cache_key = f"{url}#page{page_number - 1}"
        next_cache_key = f"{url}#page{page_number + 1}"
        
        # For now, include adjacent pages if they exist
        # This provides richer context for boundary chunks
        parts = []
        
        if prev_cache_key in self.page_cache:
            parts.append(f"[עמוד קודם {page_number - 1}]\n{self.page_cache[prev_cache_key]}")
        
        parts.append(f"[עמוד נוכחי {page_number}]\n{current_page}")
        
        if next_cache_key in self.page_cache:
            parts.append(f"[עמוד הבא {page_number + 1}]\n{self.page_cache[next_cache_key]}")
        
        return '\n\n'.join(parts)
    
    def _get_table_context(self, url: str, page_number: int, table_index: int) -> str:
        """
        Get text context surrounding a table on the page.
        
        Args:
            url: URL of the document
            page_number: Page number
            table_index: Index of the table on the page
            
        Returns:
            Context text before the table
        """
        cache_key = f"{url}#page{page_number}"
        page_items = self.page_content.get(cache_key, {})
        
        # Get text items before this table
        text_items = page_items.get('text_items', [])
        if not text_items:
            return ""
        
        # Take last 2-3 text items as context
        context_items = text_items[-3:] if len(text_items) >= 3 else text_items
        context_texts = [self._normalize_text(item.get('text', '')) for item in context_items]
        return ' '.join(context_texts)
    
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
        """Build cache of full page text and organize content by page."""
        self.page_cache = {}  # Full page context (text + tables)
        self.page_full_text = {}  # Full page text (text items only, concatenated)
        self.page_content = {}  # Organized content items by page
        
        for file_entry in self.data.get('files', []):
            url = self._normalize_url(file_entry.get('url', ''))
            content = file_entry.get('content', [])
            
            # Group content by page number
            pages_text = {}  # For text items only
            pages_all = {}   # For all content (text + tables)
            pages_items = {}  # Store actual content items
            
            for item in content:
                page_num = item.get('page_number')
                if page_num is None:
                    continue
                
                if page_num not in pages_items:
                    pages_items[page_num] = {'text_items': [], 'table_items': []}
                    
                if item.get('type') == 'text':
                    text = item.get('text', '').strip()
                    # Normalize text
                    text = self._normalize_text(text)
                    
                    if page_num not in pages_text:
                        pages_text[page_num] = []
                    if page_num not in pages_all:
                        pages_all[page_num] = []
                    
                    if text:
                        pages_text[page_num].append(text)
                        pages_all[page_num].append(text)
                        pages_items[page_num]['text_items'].append(item)
                        
                elif item.get('type') == 'table':
                    # Store table representation
                    table_text = self._table_to_text(item.get('table', {}))
                    # Normalize table text
                    table_text = self._normalize_text(table_text)
                    
                    if page_num not in pages_all:
                        pages_all[page_num] = []
                    
                    if table_text:
                        pages_all[page_num].append(f"[טבלה]\n{table_text}")
                        pages_items[page_num]['table_items'].append(item)
            
            # Store full page text (text items only, concatenated)
            for page_num, texts in pages_text.items():
                cache_key = f"{url}#page{page_num}"
                self.page_full_text[cache_key] = '\n\n'.join(texts)
            
            # Store all content for context
            for page_num, texts in pages_all.items():
                cache_key = f"{url}#page{page_num}"
                self.page_cache[cache_key] = '\n'.join(texts)
            
            # Store organized content items
            for page_num, items in pages_items.items():
                cache_key = f"{url}#page{page_num}"
                self.page_content[cache_key] = items
    
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
    
    def _create_table_chunks(self, table: Dict, context_text: str = "") -> List[tuple]:
        """
        Create chunks from table with context chunks before full table.
        Each row chunk formats as "header[0] is row[0], header[1] is row[1]".
        
        Args:
            table: Table dictionary with rows
            context_text: Surrounding text context for the table
            
        Returns:
            List of (chunk_text, chunk_type) tuples
        """
        if not table or 'rows' not in table:
            return []
        
        rows = table.get('rows', [])
        if not rows:
            return []
        
        chunks = []
        
        # Get full table text
        full_table = self._table_to_text(table)
        
        # Add 2 context chunks before full table (if context available)
        if context_text:
            # Context chunk 1: Just the context text
            chunks.append((context_text, "table_context_1"))
            
            # Context chunk 2: Context + table preview (first 2 rows)
            table_preview_rows = rows[:min(2, len(rows))]
            preview_lines = []
            for row in table_preview_rows:
                cells = row.get('cells', [])
                sorted_cells = sorted(cells, key=lambda c: c.get('col_index', 0))
                cell_texts = [self._normalize_text(cell.get('text', '')) for cell in sorted_cells]
                preview_lines.append(' | '.join(cell_texts))
            table_preview = '\n'.join(preview_lines)
            chunks.append((f"{context_text}\n\n[טבלה]\n{table_preview}", "table_context_2"))
        
        # Add full table
        chunks.append((full_table, "full"))
        
        # Extract headers
        headers = self._extract_table_headers(table)
        
        # Find first data row index (skip header rows)
        data_start_idx = 0
        for i, row in enumerate(rows):
            cells = row.get('cells', [])
            if not any(cell.get('is_header', False) for cell in cells):
                data_start_idx = i
                break
        
        # Create chunk for each data row formatted as "header is value"
        for i in range(data_start_idx, len(rows)):
            row = rows[i]
            cells = row.get('cells', [])
            sorted_cells = sorted(cells, key=lambda c: c.get('col_index', 0))
            # Normalize cell text
            cell_texts = [self._normalize_text(cell.get('text', '')) for cell in sorted_cells]
            
            # Format as "header[0] is row[0], header[1] is row[1], ..."
            pairs = []
            for j, (header, value) in enumerate(zip(headers, cell_texts)):
                if header and value:  # Only include non-empty pairs
                    pairs.append(f"ה-{header} הוא {value}")
            
            chunk_text = ', '.join(pairs)
            chunk_type = f"table_row_{i - data_start_idx + 1}"
            chunks.append((chunk_text, chunk_type))
        
        return chunks
    
    def get_pdf_segment(self, file_index: int, page_number: int) -> Optional[List[PDFSegment]]:
        """
        Get all PDF segments for a specific page in a file.
        Returns multiple segments: full page text chunks + table chunks.
        
        Args:
            file_index: Index of the file in the files array
            page_number: Page number to retrieve
            
        Returns:
            List of PDFSegment objects for the page or None if not found
        """
        try:
            file_entry = self.data['files'][file_index]
            url = self._normalize_url(file_entry.get('url', ''))
            topic = file_entry.get('topic', '')
            
            # Get all segments for this page
            segments = []
            for segment in self.iter_all_segments():
                if segment.url == url and segment.page_number == page_number:
                    segments.append(segment)
            
            return segments if segments else None
            
        except (IndexError, KeyError):
            return None
    
    def iter_all_segments(self) -> Generator[PDFSegment, None, None]:
        """
        Iterator that yields all PDF segments from all files.
        Processes full pages: extracts full page text first, then chunks pages.
        For tables, includes context chunks before full table.
        
        Yields:
            PDFSegment objects
        """
        for file_entry in self.data.get('files', []):
            topic = file_entry.get('topic', '')
            url = self._normalize_url(file_entry.get('url', ''))
            
            # Group content by page
            pages = {}
            for content_item in file_entry.get('content', []):
                page_number = content_item.get('page_number')
                if page_number is None:
                    continue
                
                if page_number not in pages:
                    pages[page_number] = {'text_items': [], 'table_items': []}
                
                if content_item.get('type') == 'text':
                    pages[page_number]['text_items'].append(content_item)
                elif content_item.get('type') == 'table':
                    pages[page_number]['table_items'].append(content_item)
            
            # Process each page
            for page_number in sorted(pages.keys()):
                page_data = pages[page_number]
                
                # Get full page context (may include adjacent pages for boundary chunks)
                cache_key = f"{url}#page{page_number}"
                entire_page_data = self.page_cache.get(cache_key, '')
                
                # Process text: concatenate all text items on page, then chunk
                text_items = page_data['text_items']
                if text_items:
                    # Get full page text (text items only)
                    full_page_text = self.page_full_text.get(cache_key, '')
                    
                    if full_page_text:
                        # Create chunks from full page text
                        chunks = self._create_chunks(full_page_text)
                        
                        # For chunks that might span pages, use cross-page context
                        for chunk_text, chunk_type in chunks:
                            # Use cross-page context for non-full chunks
                            if chunk_type != "full":
                                context = self._get_cross_page_context(url, page_number, chunk_text)
                            else:
                                context = entire_page_data
                            
                            yield PDFSegment(
                                topic=topic,
                                page_chunk=chunk_text,
                                entire_page_data=context,
                                page_number=page_number,
                                url=url,
                                chunk_type=chunk_type,
                                content_type='text'
                            )
                
                # Process tables
                table_items = page_data['table_items']
                for table_idx, table_item in enumerate(table_items):
                    table = table_item.get('table', {})
                    if not table:
                        continue
                    
                    # Get context text for this table
                    table_context = self._get_table_context(url, page_number, table_idx)
                    
                    # Create table chunks (with context chunks)
                    chunks = self._create_table_chunks(table, context_text=table_context)
                    
                    full_table_data_str = ""
                    for chunk in chunks:
                        chunk_text, chunk_type = chunk
                        full_table_data_str += chunk_text + "\n"
                    
                        
                    # Yield a segment for each chunk
                    for chunk_text, chunk_type in chunks:
                        yield PDFSegment(
                            topic=topic,
                            page_chunk=chunk_text,
                            entire_page_data=full_table_data_str, #entire_page_data,
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
        table_context_chunks = 0
        
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
                if segment.content_type == 'table':
                    if segment.chunk_type.startswith('table_row_'):
                        table_rows += 1
                    elif segment.chunk_type.startswith('table_context_'):
                        table_context_chunks += 1
            topics.add(segment.topic)
            urls.add(segment.url)
            pages.add(f"{segment.url}#page{segment.page_number}")
        
        return {
            'total_files': self.data.get('total_files', 0),
            'total_segments': total_segments,
            'text_segments': text_segments,
            'table_segments': table_segments,
            'table_context_chunks': table_context_chunks,
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
    
    # Example 1: Get first PDF page segments
    print("📄 EXAMPLE 1: Get all segments for a specific page (file_index=0, page=1)")
    print("-" * 80)
    segments = processor.get_pdf_segment(0, 1)
    if segments:
        print(f"  Found {len(segments)} segments on page 1:")
        for i, segment in enumerate(segments[:3]):  # Show first 3
            print(f"  Segment {i+1} (type: {segment.chunk_type}, content: {segment.content_type}):")
            print(f"    Topic: {segment.topic}")
            print(f"    URL: {segment.url}")
            print(f"    Page: {segment.page_number}")
            print(f"    Text: {segment.page_chunk[:100]}..." if len(segment.page_chunk) > 100 else f"    Text: {segment.page_chunk}")
            print(f"    Word count: {len(segment.page_chunk.split())} words")
            if i == 0:
                print(f"    Entire page length: {len(segment.entire_page_data)} characters")
            print()
        if len(segments) > 3:
            print(f"  ... and {len(segments) - 3} more segments")
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
    
    # Example 4: Show table processing with new formatting
    print("📊 EXAMPLE 4: Demonstrate table processing (with context and 'header is value' format)")
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
            
            # Find context chunks
            context_chunks = [s for s in page_segments 
                            if s.content_type == 'table' and s.chunk_type.startswith('table_context_')]
            
            if context_chunks:
                print(f"  Created {len(context_chunks)} context chunks before table:")
                for chunk in context_chunks:
                    print(f"    - {chunk.chunk_type}:")
                    print(f"      {chunk.page_chunk[:100]}...")
                    print()
            
            # Find row chunks with new formatting
            row_chunks = [s for s in page_segments 
                         if s.content_type == 'table' and s.chunk_type.startswith('table_row_')]
            
            if row_chunks:
                print(f"  Created {len(row_chunks)} row chunks (formatted as 'header is value'):")
                for i, chunk in enumerate(row_chunks[:2]):  # Show first 2
                    print(f"    - {chunk.chunk_type}:")
                    print(f"      {chunk.page_chunk[:150]}...")
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
    segments = processor.get_pdf_segment(0, 1)
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
