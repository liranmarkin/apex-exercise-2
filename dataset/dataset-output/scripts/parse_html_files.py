#!/usr/bin/env python3
"""
Parse HTML files from a directory structure.
Extracts h1, h2, h3, and p tags into a structured JSON output.
"""

import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from html.parser import HTMLParser
from html import unescape


class HTMLContentParser(HTMLParser):
    """Custom HTML parser to extract h1, h2, h3, and p tags."""
    
    def __init__(self):
        super().__init__()
        self.content_items = []
        self.current_tag = None
        self.current_text = []
        self.in_script = False
        self.script_id = None
        self.script_content = []
    
    def handle_starttag(self, tag, attrs):
        """Handle opening tags."""
        if tag in ['h1', 'h2', 'h3', 'p']:
            self.current_tag = tag
            self.current_text = []
        elif tag == 'script':
            # Check if this is the __NEXT_DATA__ script
            attrs_dict = dict(attrs)
            if attrs_dict.get('id') == '__NEXT_DATA__' and attrs_dict.get('type') == 'application/json':
                self.in_script = True
                self.script_content = []
    
    def handle_endtag(self, tag):
        """Handle closing tags."""
        if tag in ['h1', 'h2', 'h3', 'p'] and self.current_tag == tag:
            text = ''.join(self.current_text).strip()
            if text:  # Only add non-empty text
                self.content_items.append({
                    'tag': tag,
                    'text': text
                })
            self.current_tag = None
            self.current_text = []
        elif tag == 'script' and self.in_script:
            self.in_script = False
    
    def handle_data(self, data):
        """Handle text data within tags."""
        if self.current_tag:
            self.current_text.append(data)
        elif self.in_script:
            self.script_content.append(data)
    
    def get_next_data(self):
        """Get the parsed __NEXT_DATA__ JSON."""
        if self.script_content:
            try:
                json_str = ''.join(self.script_content)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Error parsing __NEXT_DATA__ JSON: {e}", file=sys.stderr)
        return None


def clean_html(html_text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', html_text)
    # Decode HTML entities
    clean_text = unescape(clean_text)
    # Clean up whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text


def create_url_from_path(file_path: Path) -> Optional[str]:
    """Create URL from file path, starting from www.harel-group.co.il."""
    path_str = str(file_path)
    
    # Find the index where the relevant path starts
    if 'www.harel-group.co.il' in path_str:
        start_idx = path_str.index('www.harel-group.co.il')
        url_path = path_str[start_idx:]
        
        # Ensure it starts with https://
        if not url_path.startswith('http'):
            url_path = 'https://' + url_path
        
        return url_path
    
    return None


def extract_topic_from_path(file_path: Path) -> Optional[str]:
    """Extract topic from path after www.harel-group.co.il/insurance/."""
    path_str = str(file_path)
    
    # Look for the pattern www.harel-group.co.il/insurance/XXXX
    if 'www.harel-group.co.il/insurance/' in path_str:
        # Find the part after /insurance/
        start_idx = path_str.index('www.harel-group.co.il/insurance/') + len('www.harel-group.co.il/insurance/')
        remaining_path = path_str[start_idx:]
        
        # Get the first directory/segment after /insurance/
        topic = remaining_path.split('/')[0]
        
        # If topic contains ".", take only the first part
        if '.' in topic:
            topic = topic.split('.')[0]
        
        return topic
    
    return None


def extract_text_and_strhtml(data: Any, results: List[Dict[str, Any]]) -> None:
    """Recursively extract 'text' and 'strHTML' fields from JSON data."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'text' and isinstance(value, str):
                results.append({
                    'field': 'text',
                    'original': value,
                    'clean': clean_html(value)
                })
            elif key == 'strHTML' and isinstance(value, str):
                results.append({
                    'field': 'strHTML',
                    'original': value,
                    'clean': clean_html(value)
                })
            else:
                extract_text_and_strhtml(value, results)
    elif isinstance(data, list):
        for item in data:
            extract_text_and_strhtml(item, results)


def parse_html_file(file_path: Path) -> Dict[str, Any]:
    """Parse a single HTML file and extract h1, h2, h3, and p tags."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = HTMLContentParser()
        parser.feed(html_content)
        
        # Add index to each content item
        for idx, item in enumerate(parser.content_items):
            item['index'] = idx
        
        # Extract __NEXT_DATA__ content
        next_data_fields = []
        next_data_json = parser.get_next_data()
        if next_data_json:
            extract_text_and_strhtml(next_data_json, next_data_fields)
            # Add index to next_data fields
            for idx, item in enumerate(next_data_fields):
                item['index'] = idx
        
        return {
            'file_path': str(file_path.absolute()),
            'url': create_url_from_path(file_path),
            'topic': extract_topic_from_path(file_path),
            'filename': file_path.name,
            'content': parser.content_items,
            'content_count': len(parser.content_items),
            'next_data_fields': next_data_fields,
            'next_data_count': len(next_data_fields)
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}", file=sys.stderr)
        return None


def process_directory(directory: str) -> List[Dict[str, Any]]:
    """Process all HTML files in directory and subdirectories."""
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    results = []
    html_files = list(directory_path.rglob('*.html'))
    
    print(f"Found {len(html_files)} HTML files")
    
    for html_file in html_files:
        print(f"Processing: {html_file}")
        parsed_data = parse_html_file(html_file)
        if parsed_data:
            results.append(parsed_data)
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_html_files.py <directory> [output_file]")
        print("  <directory>   : Directory containing HTML files")
        print("  [output_file] : Optional output JSON file (default: parsed_html_output.json)")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'parsed_html_output.json'
    
    print(f"Processing directory: {input_directory}")
    
    # Process all HTML files
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
