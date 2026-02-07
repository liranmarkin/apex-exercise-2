from constants import INSURANCE_TYPES
from rag.rag import RAG


def load_docs(insurance_type: str):
    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
    ]
    return docs


rag = RAG(reset_collection=True)

for insurance_type in INSURANCE_TYPES:
    docs = load_docs(insurance_type)
    rag.insert_docs(insurance_type, docs)

res = rag.query_collection("Travel", "Who is Alan Turing?")
print(res)
