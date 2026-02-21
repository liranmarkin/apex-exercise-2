import ast
import json
import time
import uuid
import asyncio
import argparse
import aiohttp


async def query_chatbot(session, question, url, model):
    """
    Sends a single question to the chatbot API (OpenAI Completions API compatible)
    and returns the assistant's reply and latency.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}]
    }

    start_time = time.time()
    try:
        async with session.post(url, json=payload) as response:
            response_data = await response.json()
    except Exception as e:
        print(f"Error querying chatbot: {e}")
        return "Error", [], time.time() - start_time

    latency = time.time() - start_time

    # Extract the answer and sources (OpenAI-like completions format)
    choice = response_data.get("choices", [{}])[0]
    answer = choice.get("text", "").strip()
    sources = choice.get("sources", [])

    return answer, sources, latency


async def process_question(sem, session, i, total, question, url, model):
    """Process a single question with concurrency limiting."""
    async with sem:
        print(f"[{i}/{total}] Querying chatbot...")
        answer, sources, bot_latency = await query_chatbot(session, question, url, model)
        print(f"[{i}/{total}] Done ({bot_latency:.2f}s)")
        return {
            "turns": [
                {
                    "role": "user",
                    "content": question,
                    "latency": 0.0
                },
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "latency": bot_latency
                }
            ],
            "metadata": {}
        }


async def run(args):
    base_url = f"http://{args.host}:{args.port}"
    url = f"{base_url.rstrip('/')}/{args.endpoint.lstrip('/')}"

    with open(args.input, "r", encoding="utf-8") as f:
        data = f.read()
        questions = ast.literal_eval(data)

    if args.limit:
        questions = questions[:args.limit]

    # Generate submission ID
    submission_id = str(uuid.uuid4()).split("-")[0]
    output_filename = f"{submission_id}_conversations.json"

    sem = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_question(sem, session, i, len(questions), q, url, args.model)
            for i, q in enumerate(questions, 1)
        ]
        conversations = await asyncio.gather(*tasks)

    # Save output (preserve original question order)
    with open(output_filename, "w", encoding="utf-8") as out:
        json.dump(list(conversations), out, ensure_ascii=False, indent=2)

    print(f"Saved {len(conversations)} conversations to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Query a local chatbot via REST API.")
    parser.add_argument("--input", required=True, help="Path to the questions TXT file")
    parser.add_argument("--host", default="127.0.0.1", help="Chatbot host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Chatbot port (default: 8000)")
    parser.add_argument("--endpoint", default="/v1/chat/completions", help="Chatbot API endpoint (default: /v1/chat/completions)")
    parser.add_argument("--model", default="best-chatbot-ever", help="Chatbot model (default: best-chatbot-ever)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of questions to evaluate")
    parser.add_argument("--concurrency", type=int, default=5, help="Max parallel requests (default: 5)")

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
