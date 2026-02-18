import sqlite3
import json
import os
from collections import defaultdict

DB_FILE = "insurance_documents.db"
OUTPUT_JSON = "insurance_export.json"


def fetch_data():
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Database not found: {DB_FILE}")

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT d.id as document_id,
               d.file_name,
               d.insurance_topic,
               d.summary,
               d.created_at,
               q.question,
               q.answer,
               q.confidence
        FROM documents d
        LEFT JOIN qa_pairs q
        ON d.id = q.document_id
        ORDER BY d.id
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def structure_data(rows):
    documents = {}

    for row in rows:
        doc_id = row["document_id"]

        if doc_id not in documents:
            documents[doc_id] = {
                "file_name": row["file_name"],
                "insurance_topic": row["insurance_topic"],
                "summary": row["summary"],
                "created_at": row["created_at"],
                "qa_pairs": []
            }

        if row["question"]:
            documents[doc_id]["qa_pairs"].append({
                "question": row["question"],
                "answer": row["answer"],
                "confidence": row["confidence"]
            })

    return documents


def print_nicely(documents):
    print("\n" + "=" * 80)
    print("INSURANCE DOCUMENT DATABASE REPORT")
    print("=" * 80 + "\n")

    total_docs = len(documents)
    total_qa = sum(len(doc["qa_pairs"]) for doc in documents.values())

    print(f"Total Documents: {total_docs}")
    print(f"Total Q&A Pairs: {total_qa}\n")

    topic_counter = defaultdict(int)

    for doc_id, doc in documents.items():
        topic_counter[doc["insurance_topic"]] += 1

        print("-" * 80)
        print(f"Document ID: {doc_id}")
        print(f"File Name: {doc['file_name']}")
        print(f"Insurance Topic: {doc['insurance_topic']}")
        print(f"Created At: {doc['created_at']}")
        print("\nSummary:")
        print(doc["summary"])
        print("\nQ&A Pairs:")

        for i, qa in enumerate(doc["qa_pairs"], 1):
            print(f"\n  {i}. Question: {qa['question']}")
            print(f"     Answer: {qa['answer']}")
            print(f"     Confidence: {qa['confidence']}")

        print("\n")

    print("=" * 80)
    print("TOPICS DISTRIBUTION")
    print("=" * 80)

    for topic, count in topic_counter.items():
        print(f"{topic}: {count}")

    print("\nDone.\n")


def export_json(documents):
    export_data = {
        "metadata": {
            "total_documents": len(documents),
            "total_qa_pairs": sum(len(doc["qa_pairs"]) for doc in documents.values())
        },
        "documents": list(documents.values())
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f"\nJSON export created: {OUTPUT_JSON}\n")


def main():
    rows = fetch_data()
    documents = structure_data(rows)
    print_nicely(documents)
    export_json(documents)


if __name__ == "__main__":
    main()
