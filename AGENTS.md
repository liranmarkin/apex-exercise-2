# Harel Insurance Customer Support Chatbot

A production-grade, domain-specific AI chatbot for Harel Insurance - Israel's largest insurance provider.

## Project Overview

This project implements an end-to-end GenAI system that:

- Ingests and structures real insurance policy data from Harel's official documentation
- Answers customer questions across **8 insurance domains**: Car, Life, Travel, Health, Dental, Mortgage, Business, and Apartment
- Grounds every answer in official documentation with explicit citations
- Aims to outperform a GPT-5 baseline using retrieval-augmented generation (RAG) and agentic system design

## Architecture

The system is built in three stages:

1. **Model Baseline & Evaluation** - Establish benchmarks and evaluation metrics
2. **Retrieval Pipeline (RAG Core)** - Document parsing, embedding, and grounded answer generation
3. **Agentic Flow & Systemization** - Multi-agent architecture with FastAPI deployment

## Tech Stack

- **Document Processing:** Docling
- **Vector DB:** Milvus
- **Agent Framework:** LangChain
- **Evaluation:** RAGAS
- **API:** FastAPI

## Project Structure

```
apex-exercise-2/
├── src/
│   ├── evaluation/      # RAGAS evaluation framework
│   └── rag/             # RAG pipeline (future)
├── dataset-raw/         # Harel insurance documents
├── scripts/             # Evaluation & utility scripts
├── tests/               # Test suite
├── plans/               # Implementation plans
└── reference-questions.json
```

## Getting Started

### 1. Download Data

Data files are hosted on a public GCS bucket. After cloning, run:

```bash
./scripts/download_data.sh
```

This downloads all dataset files (~1.4 GB) into `dataset/` and `db/`. Files that already exist locally are skipped.

### 2. Run Evaluation

Evaluate baseline (LLM-only) or RAG performance against reference questions:

```bash
uv run python scripts/evaluate_baseline.py --mode baseline --model gpt-4o
uv run python scripts/evaluate_baseline.py --mode rag
uv run python scripts/evaluate_baseline.py --mode both
uv run python scripts/evaluate_baseline.py --questions reference-questions-extended.json
```

Options:
- `--mode` - `baseline`, `rag`, or `both`
- `--model` - LLM model to use (default: `gpt-4o`)
- `--output-dir` - Results directory (default: `evaluation_results/`)
- `--questions` - Path to reference questions JSON
- `--concurrency` - Parallel LLM calls (default: 5)
- `--limit` - Max number of questions to evaluate

## Advanced Scripts

### Generate Reference Questions

Generate new question-answer pairs from the RAG dump data. Note: reference questions are already included in the repo (`reference-questions-extended.json`), so this is only needed if you want to regenerate or customize them.

```bash
uv run python scripts/generate_reference_questions.py
uv run python scripts/generate_reference_questions.py --output my-questions.json
uv run python scripts/generate_reference_questions.py --target-count 150
uv run python scripts/generate_reference_questions.py --topics apartment car health
```

Options:
- `--output` - Output file path (default: `reference-questions-extended.json`)
- `--target-count` - Target total number of questions (default: 100)
- `--topics` - Subset of topics: apartment, business, car, dental, health, life, mortgage, travel
- `--pdf-dump` / `--faq-dump` - Custom paths to dump files
- `--pdf-weight` - Proportion of PDF vs FAQ questions (default: 0.7)

### Ingest into Vector DB

Rebuild the Milvus vector database from the RAG data dumps. Note: the pre-built DB (`db/harel.db`) is already included in the data download, so this is only needed if you want to re-ingest from scratch.

```bash
uv run python scripts/ingest_rag.py
uv run python scripts/ingest_rag.py --limit 50  # limit per dump file, for testing
```

## Plans

- [Evaluation Plan](plans/evaluation.md) - RAGAS evaluation framework design

## Documentation

For complete exercise details, requirements, and scoring criteria, see [Exercise 2.md](Exercise%202.md).
