#!/usr/bin/env python3
"""
Parse Docling JSON files from a directory structure.
Extracts text, hyperlinks, and file paths into a single JSON output.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any


def decode_text(text: str) -> str:
    """Decode unicode characters in text."""
    if isinstance(text, str):
        return text.encode('utf-8').decode('unicode_escape')
    return text


def extract_content_from_json(data: Any, content_items: List[Dict[str, Any]]) -> None:
    """Recursively extract text and hyperlinks, keeping them together."""
    if isinstance(data, dict):
        # Check if this dict has both text and hyperlink
        text_value = None
        hyperlink_value = None
        
        if 'text' in data and isinstance(data['text'], str):
            text_value = data['text']
        
        for key in ['hyperlink', 'href', 'url', 'link']:
            if key in data and isinstance(data[key], str):
                hyperlink_value = data[key]
                break
        
        # If we found text, add it with its hyperlink (if any)
        if text_value:
            content_items.append({
                'text': text_value,
                'hyperlink': hyperlink_value
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


def create_url_from_path(file_path: Path) -> str:
    """Create URL from file path, starting from www.harel-group.co.il."""
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
    
    # Ensure it starts with https://
    if not url_path.startswith('http'):
        url_path = 'https://' + url_path
    
    return url_path


def parse_json_file(file_path: Path) -> Dict[str, Any]:
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
            'file_path': str(file_path.absolute()),
            'url': create_url_from_path(file_path),
            'topic': extract_topic_from_path(file_path),
            'content': content_items,
            'content_count': len(content_items)
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}", file=sys.stderr)
        return None


def process_directory(directory: str) -> List[Dict[str, Any]]:
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
        parsed_data = parse_json_file(json_file)
        if parsed_data:
            results.append(parsed_data)
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_docling_json.py <directory> [output_file]")
        print("  <directory>   : Directory containing JSON files")
        print("  [output_file] : Optional output JSON file (default: parsed_output.json)")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'parsed_output.json'
    
    print(f"Processing directory: {input_directory}")
    
    # Process all JSON files
    results = process_directory(input_directory)
    
    # Create output structure
    output_data = {
        'total_files': len(results),
        'source_directory': str(Path(input_directory).absolute()),
        'files': results
    }
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(results)} files")
    print(f"Output written to: {output_file}")


if __name__ == '__main__':
    main()
