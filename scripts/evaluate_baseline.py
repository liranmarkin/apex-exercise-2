"""
Evaluate baseline (LLM-only) and simple RAG against reference questions.

Usage:
  uv run python scripts/evaluate_baseline.py --mode baseline --model gpt-4o
  uv run python scripts/evaluate_baseline.py --mode rag
  uv run python scripts/evaluate_baseline.py --mode both
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
    args = parser.parse_args()

    modes = ["baseline", "rag"] if args.mode == "both" else [args.mode]
    for mode in modes:
        run_evaluation(mode=mode, model=args.model, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
