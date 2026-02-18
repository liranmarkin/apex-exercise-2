"""
Evaluate baseline (LLM-only) and simple RAG against reference questions.

Usage:
  uv run python scripts/evaluate_baseline.py --mode baseline --model gpt-4o
  uv run python scripts/evaluate_baseline.py --mode rag
  uv run python scripts/evaluate_baseline.py --mode both
  uv run python scripts/evaluate_baseline.py --questions reference-questions-extended.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

from evaluation.runner import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline and RAG models")
    parser.add_argument("--mode", choices=["baseline", "rag", "both"], default="baseline")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument(
        "--questions",
        default=None,
        help="Path to reference questions JSON (default: reference-questions-extended.json)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of parallel LLM calls (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of questions to evaluate",
    )
    args = parser.parse_args()

    modes = ["baseline", "rag"] if args.mode == "both" else [args.mode]
    for mode in modes:
        run_evaluation(
            mode=mode,
            model=args.model,
            output_dir=args.output_dir,
            questions_path=args.questions,
            max_concurrency=args.concurrency,
            max_questions=args.limit,
        )


if __name__ == "__main__":
    main()
