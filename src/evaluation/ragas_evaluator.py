"""
RAGAS-based evaluation for Harel Insurance chatbot.

Runs RAGAS metrics and computes competition-weighted score per the exercise rubric:
  - Relevance        65%  →  answer_relevancy
  - Citation Accuracy 15%  →  custom source matching
  - Efficiency        10%  →  latency / token cost
  - Conversational    10%  →  faithfulness (grounding proxy)
"""

import json
import os
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig

# Use the legacy import path — the "collections" metrics don't satisfy
# evaluate()'s isinstance(m, Metric) check in ragas 0.4.x.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
from ragas.metrics import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

COMPETITION_WEIGHTS = {
    "answer_relevancy": 0.65,
    "citation_accuracy": 0.15,
    "efficiency": 0.10,
    "conversational_quality": 0.10,
}


def _aggregate_scores(result) -> dict:
    """Average per-row scores from an EvaluationResult into a single dict."""
    scores = result.scores  # list[dict]
    if not scores:
        return {}
    keys = scores[0].keys()
    return {k: float(np.nanmean([s[k] for s in scores])) for k in keys}


class RAGASEvaluator:
    def __init__(self, judge_model: str = "gpt-4o", max_workers: int = 16):
        self.judge_model = judge_model
        self.max_workers = max_workers
        self.results = None

    def _build_llm(self):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=self.judge_model, temperature=0)

    @staticmethod
    def _build_embeddings():
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings()

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate_rag(self, dataset: Dataset) -> dict:
        """Run full RAGAS suite (requires contexts)."""
        metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
        run_config = RunConfig(max_workers=self.max_workers, max_retries=10, max_wait=120)
        result = evaluate(
            dataset, metrics=metrics,
            llm=self._build_llm(), embeddings=self._build_embeddings(),
            run_config=run_config,
            batch_size=self.max_workers,
        )
        self.results = result
        return _aggregate_scores(result)

    def evaluate_baseline(self, dataset: Dataset) -> dict:
        """Run only answer-level metrics (no retrieval contexts)."""
        run_config = RunConfig(max_workers=self.max_workers, max_retries=10, max_wait=120)
        result = evaluate(
            dataset, metrics=[AnswerRelevancy()],
            llm=self._build_llm(), embeddings=self._build_embeddings(),
            run_config=run_config,
            batch_size=self.max_workers,
        )
        self.results = result
        return _aggregate_scores(result)

    # ------------------------------------------------------------------
    # Competition scoring
    # ------------------------------------------------------------------

    @staticmethod
    def compute_citation_score(
        samples: list[dict],
        answers: list[str],
    ) -> float:
        """
        Basic citation accuracy: check whether the answer mentions
        the expected source file or page number.
        """
        if not samples:
            return 0.0

        hits = 0
        for sample, answer in zip(samples, answers):
            file_mentioned = sample["source_file"] in answer
            page_mentioned = str(sample["source_page"]) in answer
            if file_mentioned or page_mentioned:
                hits += 1

        return hits / len(samples)

    @staticmethod
    def compute_competition_score(
        metrics: dict,
        citation_score: float = 0.0,
        efficiency_score: float = 1.0,
    ) -> float:
        return (
            COMPETITION_WEIGHTS["answer_relevancy"] * metrics.get("answer_relevancy", 0.0)
            + COMPETITION_WEIGHTS["citation_accuracy"] * citation_score
            + COMPETITION_WEIGHTS["efficiency"] * efficiency_score
            + COMPETITION_WEIGHTS["conversational_quality"] * metrics.get("faithfulness", 0.0)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_results(
        output_path: str,
        metrics: dict,
        competition_score: float,
        per_question: Optional[list] = None,
    ) -> dict:
        output = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "competition_score": competition_score,
            "per_question_results": per_question or [],
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        return output
