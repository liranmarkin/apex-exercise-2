#!/usr/bin/env python3
"""
Parse Docling JSON files from a directory structure.
Extracts text, hyperlinks, and file paths into a single JSON output.
"""

import json
import os
import sys
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_pdf_url_mapping(csv_path: str = '_db_pdf.csv') -> Dict[str, str]:
    """Load PDF URL mapping from CSV file.
    
    Returns a dict mapping PDF filename to its real URL.
    """
    pdf_mapping = {}
    
    if not os.path.exists(csv_path):
        print(f"Warning: PDF mapping file not found: {csv_path}", file=sys.stderr)
        return pdf_mapping
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pdf_url = row.get('pdf_url', '')
                if pdf_url:
                    # Extract filename from URL (last part after /)
                    filename = pdf_url.split('/')[-1]
                    pdf_mapping[filename] = pdf_url
    except Exception as e:
        print(f"Error loading PDF mapping: {e}", file=sys.stderr)
    
    return pdf_mapping


def trim_path(path_str: str) -> str:
    """Trim everything before 'dataset-parse/data' in the path."""
    if 'dataset-parse/data' in path_str:
        start_idx = path_str.index('dataset-parse/data')
        return path_str[start_idx:]
    return path_str


def decode_text(text: str) -> str:
    """Decode unicode characters in text."""
    if isinstance(text, str):
        return text.encode('utf-8').decode('unicode_escape')
    return text


def extract_page_number(item: Dict[str, Any]) -> int:
    """Extract page number from prov field if exists."""
    if 'prov' in item and isinstance(item['prov'], list) and len(item['prov']) > 0:
        prov = item['prov'][0]
        if isinstance(prov, dict) and 'page_no' in prov:
            return prov['page_no']
    return None


def parse_table(table_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse table data into structured format with rows and columns."""
    if 'data' not in table_data or 'table_cells' not in table_data['data']:
        return None
    
    cells = table_data['data']['table_cells']
    if not cells:
        return None
    
    # Find table dimensions
    max_row = max(cell.get('end_row_offset_idx', 0) for cell in cells)
    max_col = max(cell.get('end_col_offset_idx', 0) for cell in cells)
    
    # Create a grid to hold cell data
    grid = [[None for _ in range(max_col)] for _ in range(max_row)]
    
    # Fill the grid with cell data
    for cell in cells:
        start_row = cell.get('start_row_offset_idx', 0)
        start_col = cell.get('start_col_offset_idx', 0)
        end_row = cell.get('end_row_offset_idx', start_row + 1)
        end_col = cell.get('end_col_offset_idx', start_col + 1)
        
        cell_info = {
            'text': cell.get('text', ''),
            'row_span': cell.get('row_span', 1),
            'col_span': cell.get('col_span', 1),
            'is_header': cell.get('column_header', False) or cell.get('row_header', False)
        }
        
        # Place cell in grid (only at start position for merged cells)
        if start_row < max_row and start_col < max_col:
            grid[start_row][start_col] = cell_info
    
    # Convert grid to rows format
    rows = []
    for row_idx, row in enumerate(grid):
        row_data = {
            'row_index': row_idx,
            'cells': []
        }
        for col_idx, cell in enumerate(row):
            if cell is not None:
                row_data['cells'].append({
                    'col_index': col_idx,
                    'text': cell['text'],
                    'row_span': cell['row_span'],
                    'col_span': cell['col_span'],
                    'is_header': cell['is_header']
                })
        if row_data['cells']:  # Only add non-empty rows
            rows.append(row_data)
    
    return {
        'rows': rows,
        'row_count': max_row,
        'col_count': max_col
    }


def extract_content_from_json(data: Any, content_items: List[Dict[str, Any]]) -> None:
    """Recursively extract text, hyperlinks, page numbers, and tables."""
    if isinstance(data, dict):
        # Check if this is a table
        if data.get('label') == 'table':
            table_data = parse_table(data)
            if table_data:
                content_items.append({
                    'type': 'table',
                    'table': table_data,
                    'page_number': extract_page_number(data),
                    'hyperlink': None
                })
        
        # Check if this dict has text
        text_value = None
        hyperlink_value = None
        
        if 'text' in data and isinstance(data['text'], str):
            text_value = data['text']
        
        for key in ['hyperlink', 'href', 'url', 'link']:
            if key in data and isinstance(data[key], str):
                hyperlink_value = data[key]
                break
        
        # If we found text, add it with its metadata
        if text_value:
            content_items.append({
                'type': 'text',
                'text': text_value,
                'hyperlink': hyperlink_value,
                'page_number': extract_page_number(data)
            })
        
        # Continue recursing through all values
        for value in data.values():
            extract_content_from_json(value, content_items)
    
    elif isinstance(data, list):
        for item in data:
            extract_content_from_json(item, content_items)


def get_original_file_extension(json_path: Path) -> str:
    """Check if original file is .html or .pdf on disk."""
    base_path = json_path.with_suffix('')
    
    # Check for .html first, then .pdf
    if base_path.with_suffix('.html').exists():
        return '.html'
    elif base_path.with_suffix('.pdf').exists():
        return '.pdf'
    else:
        # Default to .html if neither exists
        return '.html'


def extract_topic_from_path(file_path: Path) -> str:
    """Extract topic from path after www.harel-group.co.il/insurance/."""
    path_str = str(file_path)
    
    # Look for the pattern www.harel-group.co.il/insurance/XXXX
    if 'www.harel-group.co.il/insurance/' in path_str:
        # Find the part after /insurance/
        start_idx = path_str.index('www.harel-group.co.il/insurance/') + len('www.harel-group.co.il/insurance/')
        remaining_path = path_str[start_idx:]
        
        # Get the first directory/segment after /insurance/
        topic = remaining_path.split('/')[0]
        return topic
    
    return None


def create_url_from_path(file_path: Path, pdf_mapping: Optional[Dict[str, str]] = None) -> str:
    """Create URL from file path, starting from www.harel-group.co.il.
    
    For PDF files, looks up the real URL from the pdf_mapping dictionary.
    """
    # Get the path parts after finding 'www.harel-group.co.il' in the path
    path_str = str(file_path)
    
    # Find the index where the relevant path starts
    if 'www.harel-group.co.il' in path_str:
        start_idx = path_str.index('www.harel-group.co.il')
        url_path = path_str[start_idx:]
    else:
        # If not found, use the filename
        url_path = 'www.harel-group.co.il/' + file_path.name
    
    # Replace .json with the original extension
    original_ext = get_original_file_extension(file_path)
    url_path = url_path.replace('.json', original_ext)
    
    # For PDF files, try to find the real URL from the mapping
    if original_ext == '.pdf' and pdf_mapping:
        # Extract the PDF filename from the path
        pdf_filename = Path(url_path).name
        
        if pdf_filename in pdf_mapping:
            # Return the real PDF URL from the mapping
            return pdf_mapping[pdf_filename]
    
    # Ensure it starts with https://
    if not url_path.startswith('http'):
        url_path = 'https://' + url_path
    
    return url_path


def parse_json_file(file_path: Path, pdf_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Parse a single JSON file and extract relevant information."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract all content items (text with their hyperlinks)
        content_items = []
        extract_content_from_json(data, content_items)
        
        # Add index to each content item
        for idx, item in enumerate(content_items):
            item['index'] = idx
        
        return {
            'file_path': trim_path(str(file_path.absolute())),
            'url': create_url_from_path(file_path, pdf_mapping).replace(".html",""),
            'topic': extract_topic_from_path(file_path),
            'content': content_items,
            'content_count': len(content_items)
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}", file=sys.stderr)
        return None


def process_directory(directory: str, pdf_mapping: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Process all JSON files in directory and subdirectories."""
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    results = []
    json_files = list(directory_path.rglob('*.json'))
    
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        print(f"Processing: {json_file}")
        parsed_data = parse_json_file(json_file, pdf_mapping)
        if parsed_data:
            results.append(parsed_data)
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_docling_json.py <directory> [output_file] [pdf_csv]")
        print("  <directory>   : Directory containing JSON files")
        print("  [output_file] : Optional output JSON file (default: parsed_output.json)")
        print("  [pdf_csv]     : Optional PDF URL mapping CSV file (default: _db_pdf.csv)")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'parsed_output.json'
    pdf_csv = sys.argv[3] if len(sys.argv) > 3 else '_db_pdf.csv'
    
    print(f"Processing directory: {input_directory}")
    
    # Load PDF URL mapping
    print(f"Loading PDF URL mapping from: {pdf_csv}")
    pdf_mapping = load_pdf_url_mapping(pdf_csv)
    print(f"Loaded {len(pdf_mapping)} PDF URL mappings")
    
    # Process all JSON files
    results = process_directory(input_directory, pdf_mapping)
    
    # Create output structure
    output_data = {
        'total_files': len(results),
        'source_directory': trim_path(str(Path(input_directory).absolute())),
        'files': results
    }
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(results)} files")
    print(f"Output written to: {output_file}")


if __name__ == '__main__':
    main()
