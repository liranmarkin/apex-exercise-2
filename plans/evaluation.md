# RAGAS Evaluation Plan

## Overview
RAGAS-based evaluation framework to measure RAG pipeline quality against competition metrics.

## Competition Metrics → RAGAS Mapping

| Competition Criteria | Weight | RAGAS Metric |
|---------------------|--------|--------------|
| Relevance | 65% | `answer_relevancy` |
| Citation Accuracy | 15% | `context_precision`, `context_recall` |
| Hallucination | - | `faithfulness` |

## Configuration

- **Judge Model:** GPT-4o
- **Output Format:** JSON file (`evaluation_results.json`)

## Components

### 1. Dataset Loader (`src/evaluation/dataset.py`)
- Loads `reference-questions.json`
- Converts to RAGAS-compatible HuggingFace Dataset format
- Expected structure:
  ```python
  {
      "question": str,
      "answer": str,           # Generated answer (from RAG)
      "contexts": List[str],   # Retrieved contexts
      "ground_truth": str      # Expected answer from reference
  }
  ```

### 2. RAGAS Evaluator (`src/evaluation/ragas_evaluator.py`)
- Runs RAGAS metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- Computes weighted competition score
- Outputs results to JSON

### 3. Baseline Script (`scripts/evaluate_baseline.py`)
- Runs GPT-4o / GPT-5.2 directly (no RAG)
- Establishes baseline scores for comparison

## Integration Flow

```
Reference Questions → RAG Pipeline → RAGAS Evaluator → Competition Score
```

## Output Format

```json
{
  "timestamp": "2026-02-01T...",
  "metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.78,
    "context_precision": 0.72,
    "context_recall": 0.80
  },
  "competition_score": 0.76,
  "per_question_results": [...]
}
```

## Verification

1. `pip install -r requirements.txt`
2. `python scripts/evaluate_baseline.py`
3. Check `evaluation_results.json` for metrics
