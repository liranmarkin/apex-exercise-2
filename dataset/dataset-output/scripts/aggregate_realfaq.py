#!/usr/bin/env python3
"""
Script to aggregate all JSON files with '-realfaq.json' in their names.
Scans a directory recursively and combines all JSON content into a single output file.
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any
from html.parser import HTMLParser
from html import unescape


class TextExtractor(HTMLParser):
    """HTML parser to extract plain text from HTML content."""
    
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.in_list = False
    
    def handle_starttag(self, tag, attrs):
        # Add spacing for block elements
        if tag in ['p', 'div', 'br', 'li']:
            if tag == 'li':
                self.text_parts.append('• ')  # Add bullet for list items
                self.in_list = True
            elif tag == 'br':
                self.text_parts.append('\n')
    
    def handle_endtag(self, tag):
        # Add line breaks after block elements
        if tag in ['p', 'div', 'li', 'ul', 'ol']:
            if tag == 'li':
                self.text_parts.append('\n')
                self.in_list = False
            elif tag in ['p', 'div', 'ul', 'ol']:
                self.text_parts.append('\n')
    
    def handle_data(self, data):
        # Add the actual text content
        self.text_parts.append(data)
    
    def extract_text(self, html_content: str) -> str:
        """Extract plain text from HTML content."""
        self.text_parts = []
        self.in_list = False
        
        if not html_content:
            return ""
        
        try:
            # Decode HTML entities first
            decoded_html = unescape(html_content)
            self.feed(decoded_html)
            
            # Join all text parts and clean up
            text = ''.join(self.text_parts)
            
            # Clean up extra whitespace and line breaks
            text = re.sub(r'\n\s*\n', '\n', text)  # Multiple line breaks to single
            text = re.sub(r'[ \t]+', ' ', text)      # Multiple spaces to single
            text = text.strip()
            
            return text
            
        except Exception as e:
            print(f"Error extracting text from HTML: {e}")
            # Fallback: just remove HTML tags with regex
            return re.sub(r'<[^>]+>', '', html_content).strip()


class LinkExtractor(HTMLParser):
    """HTML parser to extract href links from HTML content."""
    
    def __init__(self):
        super().__init__()
        self.links = []
    
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr_name, attr_value in attrs:
                if attr_name == 'href' and attr_value:
                    self.links.append(attr_value)
    
    def extract_links(self, html_content: str) -> List[str]:
        """Extract all href links from HTML content."""
        self.links = []
        if html_content:
            try:
                self.feed(html_content)
            except Exception as e:
                print(f"Error parsing HTML: {e}")
        return self.links


def extract_plain_text(html_content: str) -> str:
    """
    Extract plain text from HTML content, removing all HTML tags and links.
    
    Args:
        html_content: HTML content string
        
    Returns:
        Plain text with HTML removed
    """
    if not html_content:
        return ""
    
    extractor = TextExtractor()
    return extractor.extract_text(html_content)


def clean_text(text: str) -> str:
    """
    Clean text by removing weird characters like zero-width spaces and other unwanted chars.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return text
    
    # Remove zero-width spaces and other invisible characters
    cleaned = text.replace('\u200b', '')  # Zero-width space
    cleaned = cleaned.replace('\u200c', '')  # Zero-width non-joiner
    cleaned = cleaned.replace('\u200d', '')  # Zero-width joiner
    cleaned = cleaned.replace('\u2060', '')  # Word joiner
    cleaned = cleaned.replace('\ufeff', '')  # Byte order mark
    cleaned = cleaned.replace('\u00a0', ' ')  # Non-breaking space -> regular space
    
    # Remove other common problematic characters
    cleaned = cleaned.replace('\u2028', '')  # Line separator
    cleaned = cleaned.replace('\u2029', '')  # Paragraph separator
    cleaned = cleaned.replace('\u00a0', '')
    
    # Clean up multiple spaces and trim
    #cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def extract_links_from_text(text: str) -> List[str]:
    """
    Extract href links from HTML text content.
    
    Args:
        text: HTML text content
        
    Returns:
        List of extracted href URLs
    """
    if not text:
        return []
    
    extractor = LinkExtractor()
    return extractor.extract_links(text)


def process_faq_content(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process FAQ data to extract links, clean text, and create plain text versions.
    
    Args:
        data: Original FAQ data dictionary
        
    Returns:
        Processed FAQ data with more_reference links, cleaned text, and plain text answers
    """
    if not isinstance(data, dict) or 'faqs' not in data:
        return data
    
    processed_data = data.copy()
    
    for faq in processed_data.get('faqs', []):
        if not isinstance(faq, dict):
            continue
            
        all_links = []
        
        # Extract links and clean question
        question = faq.get('question', '')
        question_links = extract_links_from_text(question)
        all_links.extend(question_links)
        faq['question'] = clean_text(question)
        
        # Extract links, clean answer, and create plain text version
        answer = faq.get('answer', '')
        answer_links = extract_links_from_text(answer)
        all_links.extend(answer_links)
        faq['answer'] = clean_text(answer)
        
        # Create plain text version of answer
        faq['answer_text'] = clean_text(extract_plain_text(answer))
        
        # Remove duplicates while preserving order
        unique_links = []
        for link in all_links:
            if link not in unique_links:
                unique_links.append(link)
        
        # Add more_reference field
        faq['more_reference'] = unique_links
    
    return processed_data


def find_realfaq_files(directory: str) -> List[Path]:
    """
    Find all files containing '-realfaq.json' in their names within the directory.
    
    Args:
        directory: Directory path to scan
        
    Returns:
        List of Path objects for matching files
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    realfaq_files = []
    for file_path in directory_path.rglob("*-realfaq.json"):
        realfaq_files.append(file_path)
    
    return realfaq_files


def read_json_file(file_path: Path) -> Dict[str, Any]:
    """
    Read and parse JSON content from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON content as dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {file_path}: {e}")
        return {}
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}


def extract_topic_from_url(source_url: str) -> str:
    """
    Extract topic from source URL.
    
    Args:
        source_url: Generated source URL
        
    Returns:
        Topic extracted from URL path
    """
    if not source_url:
        return ""
    
    # Extract topic from URL pattern: https://www.harel-group.co.il/insurance/TOPIC/...
    try:
        # Remove protocol and domain
        if "www.harel-group.co.il/insurance/" in source_url:
            # Find the part after /insurance/
            start_idx = source_url.find("/insurance/") + len("/insurance/")
            remaining_path = source_url[start_idx:]
            
            # Get the first path segment (topic)
            topic = remaining_path.split('/')[0]
            
            # Remove any file extensions
            if '.' in topic:
                topic = topic.split('.')[0]
                
            return topic
    except Exception:
        pass
    
    return ""


def generate_source_url(source_file: str) -> str:
    """
    Generate source URL based on the source file path.
    
    Args:
        source_file: Relative path of the source file
        
    Returns:
        Generated source URL
    """
    # Handle topics-faq pattern: topics-faq/XXX-faq-realfaq.json
    if source_file.startswith("topics-faq/") and "-faq-realfaq.json" in source_file:
        # Extract XXX from topics-faq/XXX-faq-realfaq.json
        filename = Path(source_file).name
        topic = filename.replace("-faq-realfaq.json", "")
        return f"https://www.harel-group.co.il/insurance/{topic}/information/faq/"
    
    # Handle paths containing www.harel-group.co.il
    if "www.harel-group.co.il" in source_file:
        # Find the substring from "www.harel-group.co.il" to "-realfaq"
        start_idx = source_file.find("www.harel-group.co.il")
        end_idx = source_file.find("-realfaq")
        
        if start_idx != -1 and end_idx != -1:
            url_part = source_file[start_idx:end_idx]
            return f"https://{url_part}.html"
    
    # Default fallback
    return ""


def aggregate_json_files(directory: str, output_file: str) -> None:
    """
    Main function to aggregate all realfaq JSON files.
    
    Args:
        directory: Directory to scan for files
        output_file: Output file path for aggregated content
    """
    # Find all realfaq files
    realfaq_files = find_realfaq_files(directory)
    
    if not realfaq_files:
        print(f"No files with '-realfaq.json' found in {directory}")
        return
    
    print(f"Found {len(realfaq_files)} realfaq files:")
    for file_path in realfaq_files:
        print(f"  - {file_path}")
    
    # Aggregate all JSON content
    aggregated_data = {
        "metadata": {
            "total_files": len(realfaq_files),
            "source_directory": str(Path(directory).resolve()),
            "files_processed": []
        },
        "content": []
    }
    
    for file_path in realfaq_files:
        print(f"Processing: {file_path}")
        json_content = read_json_file(file_path)
        
        if json_content:
            source_file_path = str(file_path.relative_to(directory))
            source_url = generate_source_url(source_file_path)
            topic = extract_topic_from_url(source_url)
            
            # Process FAQ content to extract links
            processed_content = process_faq_content(json_content)
            
            file_entry = {
                "source_file": source_file_path,
                "topic": topic,
                "source_url": source_url,
                "data": processed_content
            }
            aggregated_data["content"].append(file_entry)
            aggregated_data["metadata"]["files_processed"].append(source_file_path)
    
    # Write aggregated content to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nAggregation complete!")
        print(f"Output written to: {output_file}")
        print(f"Total files processed: {len(aggregated_data['metadata']['files_processed'])}")
        
    except Exception as e:
        print(f"Error writing output file: {e}")


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Aggregate JSON files with '-realfaq.json' in their names"
    )
    parser.add_argument(
        "directory",
        help="Directory to scan for realfaq JSON files"
    )
    parser.add_argument(
        "-o", "--output",
        default="aggregated_realfaq.json",
        help="Output file name (default: aggregated_realfaq.json)"
    )
    
    args = parser.parse_args()
    
    try:
        aggregate_json_files(args.directory, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())