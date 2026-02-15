"""
Mock RAG — implements the same interface as RAG but does nothing.

Matches the real RAG interface (insert_docs / query_collection)
so it can be swapped in anywhere the real one is used.
"""


class MockRAG:
    def __init__(self, reset_collection: bool = True):
        pass

    def insert_docs(self, insurance_type: str, docs: list[str]):
        pass

    def query_collection(self, insurance_type: str, query: str, maximal_docs: int = 2):
        return []
