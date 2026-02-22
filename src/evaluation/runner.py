"""
Evaluation runner — generates answers and evaluates them.

Keeps all orchestration logic in one place so scripts stay thin.
"""

import time
import sys
import logging
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from constants import InsuranceType

from .dataset import build_ragas_dataset, load_reference_questions
from .ragas_evaluator import RAGASEvaluator

# Import agent classes
agent_path = str(Path(__file__).parent.parent / "agent")
if agent_path not in sys.path:
    sys.path.insert(0, agent_path)

from harel_agent import HarelAgent
from response_agent import ResponseAgent

logger = logging.getLogger(__name__)

DOMAIN_TO_INSURANCE_TYPE = {
    "travel": InsuranceType.TRAVEL,
    "health": InsuranceType.HEALTH,
    "car": InsuranceType.CAR,
    "apartment": InsuranceType.APARTMENT,
    "life": InsuranceType.LIFE,
    "business": InsuranceType.BUSINESS,
    "dental": InsuranceType.DENTAL,
    "mortgage": InsuranceType.MORTGAGE,
}


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
    model: str = "openai/gpt-oss-120b",
    max_concurrency: int = 1,
) -> tuple[list[str], list[list[str]], list[float]]:
    """
    Generate baseline answers using ResponseAgent with null context.
    
    No insurance type identification or query rewriting - just raw LLM generation.
    """
    response_agent = ResponseAgent(model_name=model)
    total = len(samples)
    results = [None] * total
    pipeline_start = time.time()

    def _process(idx: int):
        sample = samples[idx]
        start = time.time()
        # Use ResponseAgent with empty RAG results (null context)
        result = response_agent.generate(
            user_input=sample['question'],
            insurance_type="UNKNOWN",  # Null type
            rag_results=[]  # No context
        )
        answer = result["answer"]
        latency = time.time() - start
        return idx, answer, [], latency

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
    model: str = "openai/gpt-oss-120b",
    max_concurrency: int = 1,
) -> tuple[list[str], list[list[str]], list[float], list[list[dict]]]:
    """
    Generate answers using HarelAgent as a blackbox.
    
    HarelAgent handles:
    - Insurance type identification
    - Query rewriting
    - RAG retrieval
    - Answer generation
    
    Returns (answers, contexts, latencies, retrieved_sources, search_queries, all_timings) where:
    - answers: list of final answers
    - contexts: list of retrieved documents (empty for HarelAgent, as it handles internally)
    - latencies: list of total latencies per question
    - retrieved_sources: list of source metadata per question
    - search_queries: list of rewritten queries per question
    - all_timings: list of timing breakdowns per question
    """
    harel_agent = HarelAgent()
    total = len(samples)
    results = [None] * total
    pipeline_start = time.time()

    def _process(idx: int):
        sample = samples[idx]
        start = time.time()
        
        # Call HarelAgent with generate_stats=True to get detailed stats
        # New signature: returns (answer_with_sources, sources_list, stats)
        answer, sources_list, stats = harel_agent.chat(sample['question'], generate_stats=True)
        
        latency = time.time() - start
        
        # Extract intermediate results from stats
        retrieved_docs = stats.get("intermediate", {}).get("retrieved_docs", [])
        rewritten_query = stats.get("intermediate", {}).get("rewritten_query", "")
        # sources_list now comes from the return value, not from stats
        sources = sources_list if sources_list else []
        
        # Build timings dict matching evaluation format
        timings = {
            "rewrite_s": stats.get("rewrite_s", 0),
            "retrieval_s": stats.get("retrieval_s", 0),
            "llm_answer_s": stats.get("llm_answer_s", 0),
            "total_s": stats.get("total_s", 0),
        }
        
        return idx, answer, retrieved_docs, sources, latency, rewritten_query, timings

    done = 0
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = {pool.submit(_process, i): i for i in range(total)}
        for future in as_completed(futures):
            idx, answer, ctx, srcs, latency, sq, timings = future.result()
            results[idx] = (answer, ctx, srcs, latency, sq, timings)
            done += 1
            if done % max(1, total // 10) == 0 or done == total:
                _print_progress(done, total, pipeline_start)

    answers = [r[0] for r in results]
    contexts = [r[1] for r in results]
    retrieved_sources = [r[2] for r in results]
    latencies = [r[3] for r in results]
    search_queries = [r[4] for r in results]
    all_timings = [r[5] for r in results]
    print(f"  Completed {total} questions in {time.time() - pipeline_start:.1f}s")
    return answers, contexts, latencies, retrieved_sources, search_queries, all_timings


# ------------------------------------------------------------------
# Full evaluation pipeline
# ------------------------------------------------------------------


def run_evaluation(
    mode: str,
    model: str = "openai/gpt-oss-120b",
    output_dir: str = "evaluation_results",
    questions_path: str | None = None,
    max_concurrency: int = 5,
    max_questions: int | None = None,
) -> dict:
    """
    End-to-end: load questions → generate answers → evaluate → save.

    mode: "baseline" | "rag"
    Returns the results dict that was saved to disk.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    samples = load_reference_questions(questions_path)
    if max_questions:
        # Sample evenly across domains instead of taking first N
        from collections import defaultdict
        by_domain = defaultdict(list)
        for s in samples:
            by_domain[s["domain"]].append(s)
        per_domain = max(1, max_questions // len(by_domain))
        sampled = []
        for domain_samples in by_domain.values():
            sampled.extend(domain_samples[:per_domain])
        samples = sampled[:max_questions]
    domains = {s["domain"] for s in samples}
    print(f"Loaded {len(samples)} questions across {len(domains)} domains: {domains}")

    # Generate answers
    retrieved_sources = None
    search_queries = None
    all_timings = None
    print(f"\n--- {mode} ({model}) ---")
    if mode == "baseline":
        answers, contexts, latencies = generate_baseline_answers(
            samples, model=model, max_concurrency=max_concurrency,
        )
    else:
        answers, contexts, latencies, retrieved_sources, search_queries, all_timings = generate_rag_answers(
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

    citation_score = evaluator.compute_citation_score(samples, answers, retrieved_sources)
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
            "latency_s": latencies[i],
        }
        if has_contexts:
            entry["contexts"] = contexts[i]
        if retrieved_sources:
            entry["retrieved_sources"] = retrieved_sources[i]
        if search_queries:
            entry["search_query"] = search_queries[i]
        if all_timings:
            entry["timings"] = all_timings[i]
        per_question.append(entry)

    # Save
    label = f"{mode}_{model.replace('.', '_')}"
    output_path = os.path.join(output_dir, f"{label}.json")
    result = evaluator.save_results(output_path, metrics, competition_score, per_question)

    print(f"\n[{label}] RAGAS metrics: {metrics}")
    print(f"[{label}] Citation accuracy: {citation_score:.4f}")
    print(f"[{label}] Efficiency score:  {efficiency_score:.4f}")
    print(f"[{label}] Competition score: {competition_score:.4f}")
    if all_timings:
        avg_identification = sum(t.get("identification_s", 0) for t in all_timings) / len(all_timings)
        avg_rewrite = sum(t.get("rewrite_s", 0) for t in all_timings) / len(all_timings)
        avg_retrieval = sum(t.get("retrieval_s", 0) for t in all_timings) / len(all_timings)
        avg_llm = sum(t.get("llm_answer_s", 0) for t in all_timings) / len(all_timings)
        print(f"[{label}] Avg timings — identification: {avg_identification:.2f}s, rewrite: {avg_rewrite:.2f}s, retrieval: {avg_retrieval:.2f}s, llm: {avg_llm:.2f}s")
    print(f"[{label}] Results saved to {output_path}")

    return result
