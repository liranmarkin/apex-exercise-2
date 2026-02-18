import os
import sys
import json
import glob
import sqlite3
import logging
import requests
from datetime import datetime

# ================= CONFIG =================

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "kimi-k2.5:cloud"
DB_FILE = "insurance_documents.db"
LOG_FILE = "processing.log"
PROCESSED_TRACK_FILE = "processed_files.txt"

# ===========================================


# ================= LOGGING =================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

# ===========================================


# ================= OLLAMA ==================

def call_ollama(prompt, expect_json=False):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)

        if response.status_code != 200:
            logging.error("Ollama returned error: %s", response.text)
            response.raise_for_status()

        return response.json()["response"]

    except Exception as e:
        logging.error("Ollama call failed: %s", str(e))
        raise


# ===========================================


# ================= SANITY CHECK ============

def sanity_check():
    logging.info("Running sanity check...")

    try:
        test_summary = """
סכם במשפט אחד:
ביטוח רכב מכסה נזק לרכב ולצד ג'.
"""
        result = call_ollama(test_summary)
        logging.info("Summary test OK: %s", result.strip())

        test_qa = """
צור 2 שאלות ותשובות קצרות בנושא ביטוח רכב.
החזר JSON בפורמט:
[
 {"question": "...", "answer": "...", "confidence": 0.9}
]
"""
        result = call_ollama(test_qa)
        json.loads(extract_json(result))
        logging.info("Q&A JSON test OK")

    except Exception:
        logging.error("Sanity check failed. Stopping execution.")
        sys.exit(1)

    logging.info("Sanity check passed.\n")


# ===========================================


# ================= DB ======================

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        insurance_topic TEXT,
        summary TEXT,
        created_at TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS qa_pairs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        question TEXT,
        answer TEXT,
        confidence REAL,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    )
    """)

    conn.commit()
    conn.close()


# ===========================================


# ================= HELPERS =================

def extract_text_from_docling(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []

    if isinstance(data, dict):
        if "text" in data:
            texts.append(data["text"])
        if "content" in data:
            texts.append(data["content"])

        for key in data:
            if isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])

    return "\n".join(texts)


def extract_json(text):
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        return text[start:end+1]
    return text


def detect_insurance_topic(text):
    prompt = f"""
זהה את סוג הביטוח המרכזי בטקסט הבא.
ענה במילה או שתיים בלבד בעברית.

טקסט:
{text[:2000]}
"""
    return call_ollama(prompt).strip()


def summarize_text(text):
    prompt = f"""
אתה מומחה ביטוח.

סכם את המסמך הבא בעברית באופן ברור ותמציתי.

מסמך:
{text[:6000]}
"""
    return call_ollama(prompt)


def generate_qa(summary):
    prompt = f"""
בהתבסס על הסיכום הבא, צור 5-10 שאלות ותשובות שמשתמשים עשויים לשאול.

החזר JSON בלבד בפורמט:
[
 {{"question": "...", "answer": "...", "confidence": 0.9}}
]

סיכום:
{summary}
"""
    response = call_ollama(prompt)
    return json.loads(extract_json(response))


# ===========================================


# ================= TRACKING =================

def load_processed():
    if not os.path.exists(PROCESSED_TRACK_FILE):
        return set()

    with open(PROCESSED_TRACK_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f.readlines())


def mark_processed(file_path):
    with open(PROCESSED_TRACK_FILE, "a", encoding="utf-8") as f:
        f.write(file_path + "\n")


# ===========================================


# ================= MAIN =====================

def process_file(json_path):
    logging.info("Processing: %s", json_path)

    text = extract_text_from_docling(json_path)

    if not text.strip():
        logging.warning("Empty text. Skipping.")
        return

    topic = detect_insurance_topic(text)
    summary = summarize_text(text)
    qa_pairs = generate_qa(summary)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
    INSERT INTO documents (file_name, insurance_topic, summary, created_at)
    VALUES (?, ?, ?, ?)
    """, (
        os.path.basename(json_path),
        topic,
        summary,
        datetime.utcnow().isoformat()
    ))

    doc_id = c.lastrowid

    for qa in qa_pairs:
        c.execute("""
        INSERT INTO qa_pairs (document_id, question, answer, confidence)
        VALUES (?, ?, ?, ?)
        """, (
            doc_id,
            qa.get("question"),
            qa.get("answer"),
            qa.get("confidence", 0.9)
        ))

    conn.commit()
    conn.close()

    mark_processed(json_path)
    logging.info("Finished: %s\n", json_path)


def main():
    if len(sys.argv) != 2:
        print("Usage: python full_build_insurance_db.py /path/to/docling_jsons")
        sys.exit(1)

    setup_logging()
    sanity_check()
    init_db()

    root_dir = sys.argv[1]
    processed = load_processed()

    json_files = glob.glob(os.path.join(root_dir, "**/*.json"), recursive=True)

    logging.info("Found %d JSON files.", len(json_files))

    for file_path in json_files:
        if file_path in processed:
            logging.info("Already processed. Skipping: %s", file_path)
            continue

        try:
            process_file(file_path)
        except Exception as e:
            logging.error("Failed processing %s: %s", file_path, str(e))

    logging.info("All done.")


if __name__ == "__main__":
    main()
