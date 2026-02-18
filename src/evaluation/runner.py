"""
Evaluation runner — generates answers and evaluates them.

Keeps all orchestration logic in one place so scripts stay thin.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from constants import InsuranceType

from .dataset import build_ragas_dataset, load_reference_questions
from .ragas_evaluator import RAGASEvaluator

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

SYSTEM_PROMPT = (
    "אתה נציג שירות לקוחות של הראל ביטוח. "
    "ענה על השאלה בצורה מדויקת ותמציתית בעברית. "
    "אם אתה מצטט מסמך, ציין את שם הקובץ ומספר העמוד."
)

QUERY_REWRITE_PROMPT = (
    "אתה מערכת חיפוש עבור חברת הראל ביטוח. "
    "קיבלת שאלה מלקוח ואתה צריך לייצר שאילתת חיפוש אופטימלית "
    "לחיפוש במאגר מסמכי ביטוח.\n\n"
    "כללים:\n"
    "- החזר רק את שאילתת החיפוש, ללא הסברים\n"
    "- השתמש במילות מפתח רלוונטיות מתחום הביטוח\n"
    "- הסר מילות שאלה מיותרות והשאר את הליבה\n"
    "- אם השאלה כוללת מונחים טכניים, שמור עליהם\n\n"
    "שאלת הלקוח: {question}\n\n"
    "שאילתת חיפוש:"
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
) -> tuple[list[str], list[list[str]], list[float], list[list[dict]]]:
    """Retrieve contexts via RAG.query_collection, then generate an answer.

    Returns (answers, contexts, latencies, retrieved_sources) where
    retrieved_sources is a list of [{url, page_index, distance}, ...] per question.
    """
    from langchain_openai import ChatOpenAI

    from rag.rag import RAG

    rag = RAG(reset_collection=False)
    llm = ChatOpenAI(model=model, temperature=0)
    total = len(samples)
    results = [None] * total
    pipeline_start = time.time()

    def _rewrite_query(question: str) -> str:
        rewrite_prompt = QUERY_REWRITE_PROMPT.format(question=question)
        response = llm.invoke(rewrite_prompt)
        return response.content.strip()

    def _process(idx: int):
        sample = samples[idx]
        start = time.time()

        search_query = _rewrite_query(sample["question"])
        insurance_type = DOMAIN_TO_INSURANCE_TYPE.get(sample["domain"])
        hits = rag.query_collection(
            insurance_type, search_query, maximal_docs=5
        )
        # Deduplicate by (url, page_index) — multiple chunks from the same
        # page can match, but we only need the full_doc once.
        seen = set()
        retrieved_docs = []
        sources = []
        for hit in (hits or []):
            url = hit["entity"].get("url", "")
            page = hit["entity"].get("page_index", -1)
            key = (url, page)
            if key in seen:
                continue
            seen.add(key)
            retrieved_docs.append(hit["entity"]["full_doc"])
            sources.append({
                "url": url,
                "page_index": page,
                "distance": hit.get("distance", 0),
            })

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
        return idx, response.content, retrieved_docs, sources, latency, search_query

    done = 0
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = {pool.submit(_process, i): i for i in range(total)}
        for future in as_completed(futures):
            idx, answer, ctx, srcs, latency, sq = future.result()
            results[idx] = (answer, ctx, srcs, latency, sq)
            done += 1
            if done % max(1, total // 10) == 0 or done == total:
                _print_progress(done, total, pipeline_start)

    answers = [r[0] for r in results]
    contexts = [r[1] for r in results]
    retrieved_sources = [r[2] for r in results]
    latencies = [r[3] for r in results]
    search_queries = [r[4] for r in results]
    print(f"  Completed {total} questions in {time.time() - pipeline_start:.1f}s")
    return answers, contexts, latencies, retrieved_sources, search_queries


# ------------------------------------------------------------------
# Full evaluation pipeline
# ------------------------------------------------------------------


def run_evaluation(
    mode: str,
    model: str = "gpt-4o",
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
    print(f"\n--- {mode} ({model}) ---")
    if mode == "baseline":
        answers, contexts, latencies = generate_baseline_answers(
            samples, model=model, max_concurrency=max_concurrency,
        )
    else:
        answers, contexts, latencies, retrieved_sources, search_queries = generate_rag_answers(
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
            "latency_s": latencies[i],
        }
        if has_contexts:
            entry["contexts"] = contexts[i]
        if retrieved_sources:
            entry["retrieved_sources"] = retrieved_sources[i]
        if search_queries:
            entry["search_query"] = search_queries[i]
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
