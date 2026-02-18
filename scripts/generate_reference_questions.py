"""
Generate reference questions from the RAG dump data (PDF segments + FAQ entries).

Scans PDF segments for factual content (amounts, percentages, time periods, exclusions,
phone numbers, age limits) and converts them into question-answer pairs with source citations.
Also includes FAQ-sourced questions to fill gaps, especially for topics without PDF data (e.g., dental).

Usage:
  python scripts/generate_reference_questions.py
  python scripts/generate_reference_questions.py --output reference-questions-extended.json
  python scripts/generate_reference_questions.py --target-count 150
  python scripts/generate_reference_questions.py --topics apartment car health
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

# Topic name mapping: English -> Hebrew
TOPIC_NAMES = {
    "apartment": "דירה",
    "business": "עסקים",
    "car": "רכב",
    "dental": "שיניים",
    "health": "בריאות",
    "life": "חיים",
    "mortgage": "משכנתא",
    "travel": "נסיעות",
}

ALL_TOPICS = list(TOPIC_NAMES.keys())

# Factual patterns to search for in PDF segments
FACTUAL_PATTERNS = [
    # Monetary amounts (NIS)
    r'\d[\d,]*\s*(?:ש"ח|שקל|שקלים)',
    # Dollar amounts
    r'\$\s*\d[\d,]*',
    r'\d[\d,]*\s*דולר',
    # Percentages
    r'\d+(?:\.\d+)?%',
    # Time periods
    r'\d+\s*(?:ימים?|חודשים?|שנים?|שנה|חודש|יום)\s*(?:רצופים?|מיום|מתאריך|ממועד)?',
    # Phone numbers
    r'(?:0\d{1,2}[-\s]?\d{7}|\*\d{4})',
    # Age limits
    r'(?:גיל|בן|בת)\s*\d+',
    # Explicit exclusions
    r'(?:לא (?:יכסה|מכסה|כולל|חל)|אינו מכוסה|למעט|פרט ל|חריג)',
]


def load_pdf_dump(dump_path: str) -> list[dict]:
    """Load PDF segments from the dump file."""
    with open(dump_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segments", [])


def load_faq_dump(dump_path: str) -> list[dict]:
    """Load FAQ entries from the dump file."""
    with open(dump_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entries", [])


def score_segment(text: str) -> int:
    """Score a text segment by how many factual patterns it contains."""
    score = 0
    for pattern in FACTUAL_PATTERNS:
        matches = re.findall(pattern, text)
        score += len(matches)
    return score


def has_clean_data(text: str) -> bool:
    """Check that a segment doesn't have too many OCR artifacts."""
    # Skip segments with long digit sequences (likely OCR garbage)
    if re.search(r"\d{6,}", text):
        return False
    # Skip very short segments
    if len(text.strip()) < 50:
        return False
    return True


def extract_factual_snippets(segments: list[dict], topics: list[str]) -> dict[str, list[dict]]:
    """Extract high-scoring factual snippets from PDF segments, grouped by topic."""
    by_topic = defaultdict(list)

    for seg in segments:
        topic = seg.get("topic", "")
        if topic not in topics:
            continue

        text = seg.get("page_chunk", "") or seg.get("entire_page_data", "")
        if not text or not has_clean_data(text):
            continue

        s = score_segment(text)
        if s >= 2:  # At least 2 factual pattern matches
            by_topic[topic].append(
                {
                    "text": text[:2000],  # Truncate very long segments
                    "page_number": seg.get("page_number", 0),
                    "url": seg.get("url", ""),
                    "score": s,
                    "topic": topic,
                }
            )

    # Sort by score descending within each topic
    for topic in by_topic:
        by_topic[topic].sort(key=lambda x: x["score"], reverse=True)

    return dict(by_topic)


def generate_questions_from_snippets(
    snippets_by_topic: dict[str, list[dict]],
    per_topic_limit: int,
) -> dict[str, list[dict]]:
    """
    Generate question-answer pairs from factual snippets.

    This uses pattern matching to create questions from the factual data found.
    For production use, consider using an LLM to generate more natural questions.
    """
    questions_by_topic = defaultdict(list)

    for topic, snippets in snippets_by_topic.items():
        hebrew_topic = TOPIC_NAMES.get(topic, topic)
        seen_pages = set()  # Avoid duplicate page references

        for snippet in snippets:
            if len(questions_by_topic[topic]) >= per_topic_limit:
                break

            page_key = (snippet["url"], snippet["page_number"])
            if page_key in seen_pages:
                continue

            text = snippet["text"]

            # Try to extract specific factual Q&A pairs
            qa_pairs = _extract_qa_from_text(text, hebrew_topic)
            for question, answer in qa_pairs:
                if len(questions_by_topic[topic]) >= per_topic_limit:
                    break
                if page_key in seen_pages:
                    continue

                questions_by_topic[topic].append(
                    {
                        "שאלה": question,
                        "תשובה": answer,
                        "מקור": {
                            "קובץ": snippet["url"],
                            "עמוד": snippet["page_number"],
                        },
                    }
                )
                seen_pages.add(page_key)

    return dict(questions_by_topic)


def _extract_qa_from_text(text: str, topic_name: str) -> list[tuple[str, str]]:
    """
    Extract question-answer pairs from a text segment using pattern matching.

    Returns a list of (question, answer) tuples.
    """
    pairs = []

    # Pattern: percentage limits (e.g., "10% מסכום הביטוח")
    pct_match = re.search(
        r"(\d+(?:\.\d+)?%)\s*(מסכום[^\n.]{5,60})", text
    )
    if pct_match:
        pct = pct_match.group(1)
        context = pct_match.group(2).strip()
        pairs.append(
            (
                f"מהו גבול האחריות ב{topic_name} עבור {context}?",
                f"{pct} {context}",
            )
        )

    # Pattern: monetary amounts (NIS)
    nis_match = re.search(
        r'(\d[\d,]+)\s*(?:ש"ח|שקלים?)\s*([^\n.]{5,60})', text
    )
    if nis_match:
        amount = nis_match.group(1)
        context = nis_match.group(2).strip()
        pairs.append(
            (
                f'מהו הסכום עבור {context} בביטוח {topic_name}?',
                f'{amount} ש"ח {context}',
            )
        )

    # Pattern: time periods
    time_match = re.search(
        r"(\d+)\s*(ימים?|חודשים?|שנים?|שנה|חודש|יום)\s*(רצופים?)?\s*([^\n.]{5,50})?",
        text,
    )
    if time_match:
        num = time_match.group(1)
        unit = time_match.group(2)
        consecutive = time_match.group(3) or ""
        context = (time_match.group(4) or "").strip()
        answer = f"{num} {unit}"
        if consecutive:
            answer += f" {consecutive}"
        if context:
            answer += f" {context}"
        pairs.append(
            (
                f"מהי תקופת הזמן הרלוונטית ל{context} בביטוח {topic_name}?",
                answer,
            )
        )

    # Pattern: phone numbers
    phone_match = re.search(r"(0\d{1,2}[-\s]?\d{7}|\*\d{4})", text)
    if phone_match:
        phone = phone_match.group(1)
        # Try to find context around the phone number
        phone_pos = text.find(phone)
        context_start = max(0, phone_pos - 60)
        context_text = text[context_start:phone_pos].strip()
        pairs.append(
            (
                f"מהו מספר הטלפון של {context_text[-40:]} בביטוח {topic_name}?",
                phone,
            )
        )

    # Pattern: exclusions
    excl_match = re.search(
        r"(לא (?:יכסה|מכסה|כולל|חל)|אינו מכוסה|למעט)\s+([^\n.]{10,80})", text
    )
    if excl_match:
        keyword = excl_match.group(1)
        what = excl_match.group(2).strip()
        pairs.append(
            (
                f"האם ביטוח {topic_name} מכסה {what}?",
                f"לא, {keyword} {what}",
            )
        )

    return pairs


def generate_faq_questions(
    faq_entries: list[dict],
    topics: list[str],
    per_topic_limit: int,
) -> dict[str, list[dict]]:
    """Generate questions from FAQ entries."""
    questions_by_topic = defaultdict(list)

    for entry in faq_entries:
        topic = entry.get("topic", "")
        if topic not in topics:
            continue
        if len(questions_by_topic[topic]) >= per_topic_limit:
            continue

        question = entry.get("question", "").strip()
        answer = entry.get("answer", "").strip()
        url = entry.get("url", "").strip()

        if not question or not answer:
            continue

        # Truncate very long answers
        if len(answer) > 500:
            answer = answer[:500] + "..."

        questions_by_topic[topic].append(
            {
                "שאלה": question,
                "תשובה": answer,
                "מקור": {"כתובת": url},
            }
        )

    return dict(questions_by_topic)


def merge_questions(
    pdf_questions: dict[str, list[dict]],
    faq_questions: dict[str, list[dict]],
    topics: list[str],
    per_topic_target: int,
) -> dict[str, list[dict]]:
    """Merge PDF and FAQ questions, preferring PDF sources, up to per-topic target."""
    merged = {}

    for topic in topics:
        hebrew_name = TOPIC_NAMES.get(topic, topic)
        pdf_qs = pdf_questions.get(topic, [])
        faq_qs = faq_questions.get(topic, [])

        combined = list(pdf_qs)  # PDF questions first
        remaining = per_topic_target - len(combined)
        if remaining > 0:
            combined.extend(faq_qs[:remaining])

        if combined:
            merged[hebrew_name] = combined

    return merged


def deduplicate(questions: list[dict]) -> list[dict]:
    """Remove duplicate questions based on question text similarity."""
    seen = set()
    unique = []
    for q in questions:
        key = q["שאלה"].strip()[:80]  # Use first 80 chars as dedup key
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference questions from RAG dump data"
    )
    parser.add_argument(
        "--output",
        default="reference-questions-extended.json",
        help="Output file path (default: reference-questions-extended.json)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=100,
        help="Target total number of questions (default: 100)",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        choices=ALL_TOPICS,
        default=ALL_TOPICS,
        help="Topics to generate questions for (default: all)",
    )
    parser.add_argument(
        "--pdf-dump",
        default="dataset/dataset-output/rag_dumps/pdf_dump.json",
        help="Path to PDF dump file",
    )
    parser.add_argument(
        "--faq-dump",
        default="dataset/dataset-output/rag_dumps/faq_dump.json",
        help="Path to FAQ dump file",
    )
    parser.add_argument(
        "--pdf-weight",
        type=float,
        default=0.7,
        help="Proportion of questions from PDF sources vs FAQ (default: 0.7)",
    )
    args = parser.parse_args()

    # Calculate per-topic targets
    num_topics = len(args.topics)
    per_topic_target = max(1, args.target_count // num_topics)
    per_topic_pdf = max(1, int(per_topic_target * args.pdf_weight))
    per_topic_faq = per_topic_target  # FAQ fills remaining slots

    print(f"Generating ~{args.target_count} questions across {num_topics} topics")
    print(f"  Per-topic target: {per_topic_target} (PDF: {per_topic_pdf}, FAQ fills rest)")
    print()

    # Load data
    print("Loading PDF dump...")
    pdf_segments = load_pdf_dump(args.pdf_dump)
    print(f"  Loaded {len(pdf_segments)} PDF segments")

    print("Loading FAQ dump...")
    faq_entries = load_faq_dump(args.faq_dump)
    print(f"  Loaded {len(faq_entries)} FAQ entries")
    print()

    # Extract factual snippets from PDFs
    print("Extracting factual snippets from PDF segments...")
    snippets_by_topic = extract_factual_snippets(pdf_segments, args.topics)
    for topic in args.topics:
        count = len(snippets_by_topic.get(topic, []))
        print(f"  {topic}: {count} factual snippets found")
    print()

    # Generate questions from PDF snippets
    print("Generating questions from PDF snippets...")
    pdf_questions = generate_questions_from_snippets(snippets_by_topic, per_topic_pdf)
    for topic in args.topics:
        count = len(pdf_questions.get(topic, []))
        print(f"  {topic}: {count} PDF questions")
    print()

    # Generate questions from FAQ
    print("Generating questions from FAQ entries...")
    faq_questions = generate_faq_questions(faq_entries, args.topics, per_topic_faq)
    for topic in args.topics:
        count = len(faq_questions.get(topic, []))
        print(f"  {topic}: {count} FAQ questions")
    print()

    # Merge and deduplicate
    print("Merging and deduplicating...")
    merged = merge_questions(pdf_questions, faq_questions, args.topics, per_topic_target)

    # Deduplicate within each topic
    total = 0
    for hebrew_name in merged:
        merged[hebrew_name] = deduplicate(merged[hebrew_name])
        count = len(merged[hebrew_name])
        total += count
        print(f"  {hebrew_name}: {count} questions")

    print(f"\nTotal: {total} questions")

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)

    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
