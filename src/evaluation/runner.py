"""
Evaluation runner — generates answers and evaluates them.

Keeps all orchestration logic in one place so scripts stay thin.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _print_progress(done: int, total: int, start_time: float):
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    print(f"  [{done}/{total}] {elapsed:.1f}s elapsed, {rate:.1f} q/s, ETA {eta:.0f}s", flush=True)


def generate_baseline_answers(
    samples: list[dict],
    model: str = "gpt-4o",
    max_concurrency: int = 1,
) -> tuple[list[str], list[list[str]], list[float]]:
    """Send each question directly to the LLM (no retrieval)."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0)
    total = len(samples)
    results = [None] * total
    pipeline_start = time.time()

    def _process(idx: int):
        sample = samples[idx]
        start = time.time()
        prompt = f"{SYSTEM_PROMPT}\n\nשאלה: {sample['question']}"
        response = llm.invoke(prompt)
        latency = time.time() - start
        return idx, response.content, [], latency

    done = 0
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = {pool.submit(_process, i): i for i in range(total)}
        for future in as_completed(futures):
            idx, answer, ctx, latency = future.result()
            results[idx] = (answer, ctx, latency)
            done += 1
            if done % max(1, total // 10) == 0 or done == total:
                _print_progress(done, total, pipeline_start)

    answers = [r[0] for r in results]
    contexts = [r[1] for r in results]
    latencies = [r[2] for r in results]
    print(f"  Completed {total} questions in {time.time() - pipeline_start:.1f}s")
    return answers, contexts, latencies


def generate_rag_answers(
    samples: list[dict],
    model: str = "gpt-4o",
    max_concurrency: int = 1,
) -> tuple[list[str], list[list[str]], list[float]]:
    """Retrieve contexts via RAG.query_collection, then generate an answer."""
    from langchain_openai import ChatOpenAI

    from rag.mock_rag import MockRAG

    rag = MockRAG(reset_collection=False)
    llm = ChatOpenAI(model=model, temperature=0)
    total = len(samples)
    results = [None] * total
    pipeline_start = time.time()

    def _process(idx: int):
        sample = samples[idx]
        start = time.time()

        hits = rag.query_collection(
            sample["domain"].capitalize(), sample["question"], maximal_docs=5
        )
        retrieved_docs = [hit["entity"]["document"] for hit in hits] if hits else []

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
        latency = time.time() - start
        return idx, response.content, retrieved_docs, latency

    done = 0
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = {pool.submit(_process, i): i for i in range(total)}
        for future in as_completed(futures):
            idx, answer, ctx, latency = future.result()
            results[idx] = (answer, ctx, latency)
            done += 1
            if done % max(1, total // 10) == 0 or done == total:
                _print_progress(done, total, pipeline_start)

    answers = [r[0] for r in results]
    contexts = [r[1] for r in results]
    latencies = [r[2] for r in results]
    print(f"  Completed {total} questions in {time.time() - pipeline_start:.1f}s")
    return answers, contexts, latencies


# ------------------------------------------------------------------
# Full evaluation pipeline
# ------------------------------------------------------------------


def run_evaluation(
    mode: str,
    model: str = "gpt-4o",
    output_dir: str = "evaluation_results",
    questions_path: str | None = None,
    max_concurrency: int = 5,
) -> dict:
    """
    End-to-end: load questions → generate answers → evaluate → save.

    mode: "baseline" | "rag"
    Returns the results dict that was saved to disk.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    samples = load_reference_questions(questions_path)
    domains = {s["domain"] for s in samples}
    print(f"Loaded {len(samples)} questions across {len(domains)} domains: {domains}")

    # Generate answers
    print(f"\n--- {mode} ({model}) ---")
    if mode == "baseline":
        answers, contexts, latencies = generate_baseline_answers(
            samples, model=model, max_concurrency=max_concurrency,
        )
    else:
        answers, contexts, latencies = generate_rag_answers(
            samples, model=model, max_concurrency=max_concurrency,
        )

    # Build RAGAS dataset
    dataset = build_ragas_dataset(samples, answers, contexts)

    # Evaluate
    evaluator = RAGASEvaluator(max_workers=max_concurrency)
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
