from pymilvus import MilvusClient, DataType, model

from constants import DB_PATH, COLLECTION_NAME


class RAG:
    def __init__(self, reset_collection: bool = True):
        self.embeder = self._get_embeder()
        self.collection = COLLECTION_NAME
        self.client = self._get_db_client(DB_PATH)
        self.schema = self._get_schema()
        
        if reset_collection:
            self._reset_collection()

        self._create_indices()

    @staticmethod
    def _get_embeder():
        return model.DefaultEmbeddingFunction()

    def _get_db_client(self, db_path: str):
        client = MilvusClient(db_path)
        return client
    
    def _get_schema(self):
        schema = self.client.create_schema()
        schema.add_field(field_name="id", is_primary=True, auto_id=True, datatype=DataType.INT64)
        schema.add_field(field_name="embed", datatype=DataType.FLOAT_VECTOR, dim=self.embeder.dim)
        schema.add_field(field_name="insurance_type", datatype=DataType.VARCHAR, max_length=50)
        # TODO: updata document datatype
        schema.add_field(field_name="document", datatype=DataType.VARCHAR, max_length=1000)
        return schema

    def _reset_collection(self):
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        self.client.create_collection(self.collection, schema=self.schema)

    def _create_indices(self):
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embed", metric_type="COSINE")
        self.client.create_index(self.collection, index_params)

    def insert_docs(self, insurance_type: str, docs: list[str]):
        embeds = self.embeder.encode_documents(docs)
        data = [{"embed": embeds[i], "insurance_type": insurance_type, "document": docs[i]} for i in range(len(docs))]
        res = self.client.insert(collection_name=self.collection, data=data)
        return res

    def query_collection(self, insurance_type: str, query: str, maximal_docs: int = 2):
        vectors = self.embeder.encode_queries([query])
        filter = f"insurance_type == '{insurance_type}'"
        res = self.client.search(collection_name=self.collection, data=vectors, filter=filter, output_fields=["document"], limit=maximal_docs)[0]
        return res
