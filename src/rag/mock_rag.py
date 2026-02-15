"""
Simple RAG stub — implements the interface but does no retrieval.

This is the dumbest possible baseline: just passes the question
straight to the LLM with no context. Exists so the evaluation
pipeline has a consistent interface to call.
"""

from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = (
    "אתה נציג שירות לקוחות של הראל ביטוח. "
    "ענה על השאלה בצורה מדויקת ותמציתית בעברית. "
    "אם אתה מצטט מסמך, ציין את שם הקובץ ומספר העמוד."
)


class MockRAG:
    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model, temperature=0)

    def retrieve(self, question: str, domain: str = "", top_k: int = 5) -> list[str]:
        return []

    def answer(self, question: str, domain: str = "") -> tuple[str, list[str]]:
        """Return (generated_answer, retrieved_contexts)."""
        contexts = self.retrieve(question, domain)
        prompt = f"{SYSTEM_PROMPT}\n\nשאלה: {question}"
        response = self.llm.invoke(prompt)
        return response.content, contexts
