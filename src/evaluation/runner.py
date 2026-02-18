"""
Evaluation runner — generates answers and evaluates them.

Keeps all orchestration logic in one place so scripts stay thin.
"""

import time

from .dataset import build_ragas_dataset, load_reference_questions
from .ragas_evaluator import RAGASEvaluator

SYSTEM_PROMPT = (
    "אתה נציג שירות לקוחות של הראל ביטוח. "
    "ענה על השאלה בצורה מדויקת ותמציתית בעברית. "
    "אם אתה מצטט מסמך, ציין את שם הקובץ ומספר העמוד."
)


# ------------------------------------------------------------------
# Answer generators
# ------------------------------------------------------------------


def generate_baseline_answers(
    samples: list[dict],
    model: str = "gpt-4o",
) -> tuple[list[str], list[list[str]], list[float]]:
    """Send each question directly to the LLM (no retrieval)."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0)

    answers, contexts, latencies = [], [], []
    for sample in samples:
        start = time.time()
        prompt = f"{SYSTEM_PROMPT}\n\nשאלה: {sample['question']}"
        response = llm.invoke(prompt)
        latencies.append(time.time() - start)
        answers.append(response.content)
        contexts.append([])
    return answers, contexts, latencies


def generate_rag_answers(
    samples: list[dict],
    model: str = "gpt-4o",
) -> tuple[list[str], list[list[str]], list[float]]:
    """Retrieve contexts via RAG.query_collection, then generate an answer."""
    from langchain_openai import ChatOpenAI

    from rag.mock_rag import MockRAG

    rag = MockRAG(reset_collection=False)
    llm = ChatOpenAI(model=model, temperature=0)

    answers, contexts, latencies = [], [], []
    for sample in samples:
        start = time.time()

        results = rag.query_collection(
            sample["domain"].capitalize(), sample["question"], maximal_docs=5
        )
        retrieved_docs = [hit["entity"]["document"] for hit in results] if results else []
        contexts.append(retrieved_docs)

        if retrieved_docs:
            context_str = "\n\n".join(retrieved_docs)
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"ענה בהתבסס אך ורק על המידע הבא:\n{context_str}\n\n"
                f"שאלה: {sample['question']}"
            )
        else:
            prompt = f"{SYSTEM_PROMPT}\n\nשאלה: {sample['question']}"

        response = llm.invoke(prompt)
        latencies.append(time.time() - start)
        answers.append(response.content)
    return answers, contexts, latencies


# ------------------------------------------------------------------
# Full evaluation pipeline
# ------------------------------------------------------------------


def run_evaluation(
    mode: str,
    model: str = "gpt-4o",
    output_dir: str = "evaluation_results",
) -> dict:
    """
    End-to-end: load questions → generate answers → evaluate → save.

    mode: "baseline" | "rag"
    Returns the results dict that was saved to disk.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    samples = load_reference_questions()
    domains = {s["domain"] for s in samples}
    print(f"Loaded {len(samples)} questions across {len(domains)} domains: {domains}")

    # Generate answers
    print(f"\n--- {mode} ({model}) ---")
    if mode == "baseline":
        answers, contexts, latencies = generate_baseline_answers(samples, model=model)
    else:
        answers, contexts, latencies = generate_rag_answers(samples, model=model)

    # Build RAGAS dataset
    dataset = build_ragas_dataset(samples, answers, contexts)

    # Evaluate
    evaluator = RAGASEvaluator()
    has_contexts = any(len(c) > 0 for c in contexts)
    if has_contexts:
        metrics = evaluator.evaluate_rag(dataset)
    else:
        metrics = evaluator.evaluate_baseline(dataset)

    citation_score = evaluator.compute_citation_score(samples, answers)
    avg_latency = sum(latencies) / len(latencies)
    efficiency_score = max(0.0, 1.0 - avg_latency / 30.0)
    competition_score = evaluator.compute_competition_score(
        metrics, citation_score, efficiency_score
    )

    # Per-question breakdown
    per_question = []
    for i, sample in enumerate(samples):
        entry = {
            "question": sample["question"],
            "domain": sample["domain"],
            "ground_truth": sample["ground_truth"],
            "generated_answer": answers[i],
            "source_file": sample["source_file"],
            "source_page": sample["source_page"],
            "latency_s": latencies[i],
        }
        if has_contexts:
            entry["contexts"] = contexts[i]
        per_question.append(entry)

    # Save
    label = f"{mode}_{model.replace('.', '_')}"
    output_path = os.path.join(output_dir, f"{label}.json")
    result = evaluator.save_results(output_path, metrics, competition_score, per_question)

    print(f"\n[{label}] RAGAS metrics: {metrics}")
    print(f"[{label}] Citation accuracy: {citation_score:.4f}")
    print(f"[{label}] Efficiency score:  {efficiency_score:.4f}")
    print(f"[{label}] Competition score: {competition_score:.4f}")
    print(f"[{label}] Results saved to {output_path}")

    return result
