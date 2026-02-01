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

## Plans

- [Evaluation Plan](plans/evaluation.md) - RAGAS evaluation framework design

## Documentation

For complete exercise details, requirements, and scoring criteria, see [Exercise 2.md](Exercise%202.md).
