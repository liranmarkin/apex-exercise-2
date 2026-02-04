from pymilvus import MilvusClient, model

from constants import DB_PATH, COLLECTION_NAME


class RAG:
    def __init__(self, reset_collection: bool = True):
        self.embed = self._get_embeder()
        self.collection = COLLECTION_NAME
        self.client = self._get_db_client(DB_PATH)
        
        if reset_collection:
            self._reset_collection()

    @staticmethod
    def _get_embeder():
        return model.DefaultEmbeddingFunction()

    def _get_db_client(self, db_path: str):
        client = MilvusClient(db_path)
        return client

    def _reset_collection(self):
        if self.client.has_collection(collection_name=self.collection):
            self.client.drop_collection(collection_name=self.collection)
        self.client.create_collection(collection_name=self.collection, dimension=self.embed.dim)

    def insert_docs(self, insurance_type: str, docs: list[str]):
        vectors = self.embed.encode_documents(docs)
        data = [{"id": i, "vector": vectors[i], "text": docs[i], "subject": insurance_type} for i in range(len(docs))]
        res = self.client.insert(collection_name=self.collection, data=data)
        return res

    def query_collection(self, query: str, maximal_docs: int = 2):
        vectors = self.embed.encode_queries([query])
        # TODO: filter insurance type
        res = self.client.search(collection_name=self.collection, data=vectors, output_fields=["subject", "text"], limit=maximal_docs)[0]
        return res
