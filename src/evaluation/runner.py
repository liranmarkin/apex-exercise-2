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
    "אתה נציג שירות לקוחות מומחה של הראל ביטוח. "
    "תפקידך לענות על שאלות לקוחות בצורה מדויקת, מקצועית ומועילה בעברית.\n\n"
    "הנחיות:\n"
    "1. ענה על בסיס המידע שסופק לך. אם המידע מכיל תשובה חלקית, ספק את מה שאתה יכול.\n"
    "2. ציין את המקור: כתובת URL ומספר עמוד אם זמין.\n"
    "3. אם המידע שסופק לא מכיל תשובה ישירה, נסה להסיק תשובה מהמידע הזמין.\n"
    "4. היה תמציתי אך מלא - אל תשמיט פרטים חשובים."
)

QUERY_REWRITE_PROMPT = (
    "אתה מערכת חיפוש עבור חברת הראל ביטוח. "
    "קיבלת שאלה מלקוח ואתה צריך לייצר שאילתת חיפוש אופטימלית "
    "לחיפוש במאגר מסמכי ביטוח.\n\n"
    "כללים:\n"
    "- החזר רק את שאילתת החיפוש, ללא הסברים\n"
    "- השתמש במילות מפתח רלוונטיות מתחום הביטוח\n"
    "- הסר מילות שאלה מיותרות והשאר את הליבה\n"
    "- אם השאלה כוללת מונחים טכניים, שמור עליהם\n"
    "- כלול מונחים נרדפים רלוונטיים\n\n"
    "דוגמאות:\n"
    "שאלה: מתי דירה נחשבת לא תפוסה לפי פוליסת ביטוח דירה?\n"
    "שאילתת חיפוש: דירה לא תפוסה פנויה תנאי פוליסה ימים רצופים\n\n"
    "שאלה: האם ביטוח הנסיעות מכסה ספורט אתגרי?\n"
    "שאילתת חיפוש: ביטוח נסיעות ספורט אתגרי כיסוי חריגים פעילות\n\n"
    "שאלה: מה מספר הטלפון של מוקד התביעות של הראל?\n"
    "שאילתת חיפוש: מוקד תביעות הראל טלפון מספר פנייה\n\n"
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

    llm = ChatOpenAI(model=model, temperature=0, max_retries=5)
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
    # Warm up Milvus index to avoid cold-start latency on the first real query
    _warmup_vec = rag.embeder.encode_queries(["warmup"])[0]
    rag.client.search(collection_name=rag.collection, data=[_warmup_vec], limit=1)
    llm = ChatOpenAI(model=model, temperature=0, max_retries=5)
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

        t0 = time.time()
        search_query = _rewrite_query(sample["question"])
        rewrite_time = time.time() - t0

        insurance_type = DOMAIN_TO_INSURANCE_TYPE.get(sample["domain"])
        t0 = time.time()
        hits = rag.query_collection(
            insurance_type, search_query, maximal_docs=15
        )
        retrieval_time = time.time() - t0

        # Deduplicate by (url, page_index), keeping the best distance per key
        seen = {}
        for hit in (hits or []):
            url = hit["entity"].get("url", "")
            page = hit["entity"].get("page_index", -1)
            key = (url, page)
            dist = hit.get("distance", 0)
            if key not in seen or dist > seen[key].get("distance", 0):
                seen[key] = {
                    "entity": hit["entity"],
                    "distance": dist,
                }

        # Score, filter, and rank — take top 5
        MIN_DISTANCE = 0.45  # reject low-similarity results
        FAQ_BOOST = 1.15
        candidates = []
        for key, hit in seen.items():
            dist = hit["distance"]
            if dist < MIN_DISTANCE:
                continue
            source_type = hit["entity"].get("source_type", 2)
            score = dist * FAQ_BOOST if source_type == 1 else dist  # FAQ boost
            candidates.append((score, key, hit))
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:5]

        retrieved_docs = []
        sources = []
        for score, key, hit in top:
            retrieved_docs.append(hit["entity"]["full_doc"])
            sources.append({
                "url": hit["entity"].get("url", ""),
                "page_index": hit["entity"].get("page_index", -1),
                "distance": hit["distance"],
                "source_type": hit["entity"].get("source_type", 2),
            })

        if retrieved_docs:
            context_parts = []
            for doc, src in zip(retrieved_docs, sources):
                url = src.get("url", "")
                page = src.get("page_index", -1)
                source_label = f"[מקור: {url}"
                if page > 0:
                    source_label += f", עמוד {page}"
                source_label += "]"
                context_parts.append(f"{source_label}\n{doc}")
            context_str = "\n\n---\n\n".join(context_parts)
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"להלן מידע ממסמכי הראל ביטוח הרלוונטיים לשאלה:\n\n{context_str}\n\n"
                f"שאלת הלקוח: {sample['question']}\n\n"
                f"ענה על בסיס המידע לעיל. ציין את המקור (URL ועמוד) בסוף התשובה."
            )
        else:
            prompt = f"{SYSTEM_PROMPT}\n\nשאלה: {sample['question']}"

        t0 = time.time()
        response = llm.invoke(prompt)
        llm_time = time.time() - t0

        latency = time.time() - start
        timings = {
            "rewrite_s": round(rewrite_time, 3),
            "retrieval_s": round(retrieval_time, 3),
            "llm_answer_s": round(llm_time, 3),
            "total_s": round(latency, 3),
        }
        return idx, response.content, retrieved_docs, sources, latency, search_query, timings

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
        avg_rewrite = sum(t["rewrite_s"] for t in all_timings) / len(all_timings)
        avg_retrieval = sum(t["retrieval_s"] for t in all_timings) / len(all_timings)
        avg_llm = sum(t["llm_answer_s"] for t in all_timings) / len(all_timings)
        print(f"[{label}] Avg timings — rewrite: {avg_rewrite:.2f}s, retrieval: {avg_retrieval:.2f}s, llm: {avg_llm:.2f}s")
    print(f"[{label}] Results saved to {output_path}")

    return result
