#!/usr/bin/env python3
"""
Unified script to dump all datasets for RAG system.
Provides 4 dump functions for different data sources.
"""

import json
import sys
import os
from pathlib import Path

# Add current directory to path to import the processors
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_policies_for_rag import PoliciesDataProcessor
from prepare_pdf_for_rag import PDFDataProcessor
from prepare_faq_for_rag import FAQDataExtractor


def dump_policies_data(json_path: str, output_path: str = None, chunk_threshold: int = 50):
    """
    Dump policies (HTML) data to JSON file.
    
    Args:
        json_path: Path to topics-policies-manual-html-parse.json
        output_path: Output JSON file path (default: policies_dump.json)
        chunk_threshold: Word count threshold for chunking
        
    Returns:
        List of policy segments
    """
    print("=" * 80)
    print("DUMPING POLICIES DATA (HTML)")
    print("=" * 80)
    
    processor = PoliciesDataProcessor(json_path, chunk_threshold=chunk_threshold)
    
    # Get statistics
    stats = processor.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Total strHTML segments: {stats['total_strhtml_segments']}")
    print(f"   Total chunks (with overlaps): {stats['total_chunks_including_overlaps']}")
    print(f"   Topics: {', '.join(stats['topics_list'])}")
    
    # Extract all segments
    print(f"\n🔄 Extracting all segments...")
    segments = []
    for segment in processor.iter_all_segments():
        segments.append(segment.to_dict())
    
    print(f"✅ Extracted {len(segments)} segments")
    
    # Prepare output
    output_data = {
        'source': 'policies-html',
        'statistics': stats,
        'segments': segments
    }
    
    # Write to file if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved to: {output_path}")
    
    print("=" * 80)
    print()
    
    return segments


def dump_pdf_data(json_path: str, output_path: str = None, chunk_threshold: int = 50):
    """
    Dump PDF (Docling) data to JSON file.
    
    Args:
        json_path: Path to topics-pdf-docling.json
        output_path: Output JSON file path (default: pdf_dump.json)
        chunk_threshold: Word count threshold for chunking
        
    Returns:
        List of PDF segments
    """
    print("=" * 80)
    print("DUMPING PDF DATA (Docling)")
    print("=" * 80)
    
    processor = PDFDataProcessor(json_path, chunk_threshold=chunk_threshold)
    
    # Get statistics
    stats = processor.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Total segments: {stats['total_segments']}")
    print(f"   - Text segments: {stats['text_segments']}")
    print(f"   - Table segments: {stats['table_segments']}")
    print(f"   Table row chunks: {stats['table_row_chunks']}")
    print(f"   Total chunks (with overlaps): {stats['total_chunks_including_overlaps']}")
    print(f"   Topics: {', '.join(stats['topics_list'])}")
    
    # Extract all segments
    print(f"\n🔄 Extracting all segments...")
    segments = []
    for segment in processor.iter_all_segments():
        segments.append(segment.to_dict())
    
    print(f"✅ Extracted {len(segments)} segments")
    
    # Prepare output
    output_data = {
        'source': 'pdf-docling',
        'statistics': stats,
        'segments': segments
    }
    
    # Write to file if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved to: {output_path}")
    
    print("=" * 80)
    print()
    
    return segments


def dump_faq_data(json_path: str, output_path: str = None):
    """
    Dump FAQ data to JSON file.
    
    Args:
        json_path: Path to topics-faq.json
        output_path: Output JSON file path (default: faq_dump.json)
        
    Returns:
        List of FAQ entries
    """
    print("=" * 80)
    print("DUMPING FAQ DATA")
    print("=" * 80)
    
    extractor = FAQDataExtractor(json_path)
    
    # Get statistics
    total_count = extractor.get_total_count()
    topics = extractor.get_topics()
    
    print(f"\n📊 Statistics:")
    print(f"   Total FAQ entries: {total_count}")
    print(f"   Topics: {', '.join(topics)}")
    
    # Extract all entries
    print(f"\n🔄 Extracting all FAQ entries...")
    entries = extractor.extract_faq_entries()
    
    print(f"✅ Extracted {len(entries)} FAQ entries")
    
    # Prepare output
    output_data = {
        'source': 'faq',
        'statistics': {
            'total_entries': total_count,
            'topics': topics
        },
        'entries': entries
    }
    
    # Write to file if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved to: {output_path}")
    
    print("=" * 80)
    print()
    
    return entries


def dump_all_data(blogs_path: str, policies_path: str, pdf_path: str, faq_path: str, 
                  output_dir: str = "output", chunk_threshold: int = 50):
    """
    Dump all datasets to separate JSON files.
    
    Args:
        policies_path: Path to topics-policies-manual-html-parse.json
        pdf_path: Path to topics-pdf-docling.json
        faq_path: Path to topics-faq.json
        output_dir: Directory to save output files
        chunk_threshold: Word count threshold for chunking
        
    Returns:
        Dictionary with all data
    """
    print("=" * 80)
    print("DUMPING ALL DATASETS")
    print("=" * 80)
    print()
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Dump each dataset
    policies_output = os.path.join(output_dir, "policies_dump.json")
    blogs_output = os.path.join(output_dir, "blogs_dump.json")
    pdf_output = os.path.join(output_dir, "pdf_dump.json")
    faq_output = os.path.join(output_dir, "faq_dump.json")
    
    policies_data = dump_policies_data(policies_path, policies_output, chunk_threshold)
    blogs_data = dump_policies_data(blogs_path, blogs_output, chunk_threshold)
    pdf_data = dump_pdf_data(pdf_path, pdf_output, chunk_threshold)
    faq_data = dump_faq_data(faq_path, faq_output)
    
    # Create combined summary
    summary = {
        'total_segments': len(policies_data) + len(pdf_data) + len(faq_data),
        'policies_segments': len(policies_data),
        'pdf_segments': len(pdf_data),
        'faq_entries': len(faq_data),
        'output_files': {
            'policies': policies_output,
            'blogs': blogs_output,
            'pdf': pdf_output,
            'faq': faq_output
        }
    }
    
    # Save summary
    summary_output = os.path.join(output_dir, "dump_summary.json")
    with open(summary_output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("=" * 80)
    print("📊 COMBINED SUMMARY")
    print("=" * 80)
    print(f"   Total segments across all sources: {summary['total_segments']}")
    print(f"   - Policies (HTML): {summary['policies_segments']}")
    print(f"   - PDF (Docling): {summary['pdf_segments']}")
    print(f"   - FAQ: {summary['faq_entries']}")
    print()
    print(f"💾 Output files saved to: {output_dir}/")
    print(f"   - {os.path.basename(policies_output)}")
    print(f"   - {os.path.basename(blogs_output)}")
    print(f"   - {os.path.basename(pdf_output)}")
    print(f"   - {os.path.basename(faq_output)}")
    print(f"   - {os.path.basename(summary_output)}")
    print("=" * 80)
    print()
    
    return {
        'policies': policies_data,
        'pdf': pdf_data,
        'faq': faq_data,
        'summary': summary
    }


def main():
    """Main entry point with CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Dump all datasets for RAG system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dump all datasets
  python dump_all_datasets.py --all \\
    --policies dataset/topics-policies-manual-html-parse.json \\
    --pdf dataset/topics-pdf-docling.json \\
    --faq dataset/topics-faq.json \\
    --output-dir output

  # Dump only policies
  python dump_all_datasets.py --policies dataset/topics-policies-manual-html-parse.json

  # Dump only blogs
  python dump_all_datasets.py --policies dataset/topic-information-blogs-manual-html-parse.json

  # Dump only PDF with custom threshold
  python dump_all_datasets.py --pdf dataset/topics-pdf-docling.json --chunk-threshold 30

  # Dump only FAQ
  python dump_all_datasets.py --faq dataset/topics-faq.json
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Dump all datasets (requires --policies, --blogs, --pdf, --faq)')
    parser.add_argument('--blogs', type=str,
                       help='Path to topic-information-blogs-manual-html-parse.json')
    parser.add_argument('--policies', type=str,
                       help='Path to topics-policies-manual-html-parse.json')
    parser.add_argument('--pdf', type=str,
                       help='Path to topics-pdf-docling.json')
    parser.add_argument('--faq', type=str,
                       help='Path to topics-faq.json')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for dump files (default: output)')
    parser.add_argument('--chunk-threshold', type=int, default=50,
                       help='Word count threshold for chunking (default: 50)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.all:
        if not all([args.blogs, args.policies, args.pdf, args.faq]):
            parser.error("--all requires --policies, --blogs, --pdf, and --faq")
        dump_all_data(args.blogs, args.policies, args.pdf, args.faq, 
                     args.output_dir, args.chunk_threshold)
    elif args.blogs:
        output_path = os.path.join(args.output_dir, "blogs_dump.json")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        dump_policies_data(args.policies, output_path, args.chunk_threshold)
    elif args.policies:
        output_path = os.path.join(args.output_dir, "policies_dump.json")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        dump_policies_data(args.policies, output_path, args.chunk_threshold)
    elif args.pdf:
        output_path = os.path.join(args.output_dir, "pdf_dump.json")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        dump_pdf_data(args.pdf, output_path, args.chunk_threshold)
    elif args.faq:
        output_path = os.path.join(args.output_dir, "faq_dump.json")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        dump_faq_data(args.faq, output_path)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
