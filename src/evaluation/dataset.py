"""
Dataset loader for RAGAS evaluation.

Loads reference questions JSON and converts to evaluation-ready format.
Supports two source formats:
  - PDF sources:  { "שאלה": str, "תשובה": str, "מקור": { "קובץ": str, "עמוד": int } }
  - FAQ sources:  { "שאלה": str, "תשובה": str, "מקור": { "כתובת": str } }
"""

import json
import os
from typing import Optional

from datasets import Dataset

# Hebrew → English domain mapping
DOMAIN_MAP = {
    "דירה": "apartment",
    "עסקים": "business",
    "רכב": "car",
    "בריאות": "health",
    "נסיעות": "travel",
    "חיים": "life",
    "שיניים": "dental",
    "משכנתא": "mortgage",
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REFERENCE_QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "reference-questions-extended.json")


def load_reference_questions(path: Optional[str] = None) -> list[dict]:
    """
    Load reference questions and flatten into evaluation samples.

    Each sample:
      {
        "question": str,
        "ground_truth": str,
        "domain": str,           # English
        "domain_he": str,        # Hebrew
        "source_file": str,
        "source_page": int,
      }
    """
    path = path or REFERENCE_QUESTIONS_PATH

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for domain_he, questions in data.items():
        domain_en = DOMAIN_MAP.get(domain_he, domain_he)
        for q in questions:
            source = q.get("מקור", {})
            samples.append({
                "question": q["שאלה"],
                "ground_truth": q["תשובה"],
                "domain": domain_en,
                "domain_he": domain_he,
                "source_file": source.get("קובץ", source.get("כתובת", "")),
                "source_page": source.get("עמוד", 0),
            })

    return samples


def build_ragas_dataset(
    samples: list[dict],
    answers: list[str],
    contexts: list[list[str]],
) -> Dataset:
    """
    Combine evaluation samples with generated answers/contexts
    into a RAGAS-compatible HuggingFace Dataset.
    """
    return Dataset.from_dict({
        "question": [s["question"] for s in samples],
        "answer": answers,
        "contexts": contexts,
        "ground_truth": [s["ground_truth"] for s in samples],
    })
