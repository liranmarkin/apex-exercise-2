from typing import Generator

from pymilvus import MilvusClient, DataType, model

from constants import DB_PATH, COLLECTION_NAME, InsuranceType


class RAG:
    def __init__(self, reset_collection: bool = False):
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
        schema.add_field(field_name="embeding", datatype=DataType.FLOAT_VECTOR, dim=self.embeder.dim)
        schema.add_field(field_name="insurance_type", datatype=DataType.INT8)
        schema.add_field(field_name="full_doc", datatype=DataType.VARCHAR, max_length=10000)
        schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="page_index", datatype=DataType.INT8)
        schema.add_field(field_name="hyperlinks", datatype=DataType.JSON)
        return schema

    def _reset_collection(self):
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        self.client.create_collection(self.collection, schema=self.schema)

    def _create_indices(self):
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embeding", metric_type="COSINE")
        self.client.create_index(self.collection, index_params)

    def insert_doc(self, chunk: str, insurance_type: InsuranceType, full_doc: str, url: str, page_index: int = -1, hyperlinks: dict[str, str] = dict()):
        embeding = self.embeder.encode_documents([chunk])[0]
        data = [{
            "embeding": embeding,
            "insurance_type": insurance_type.value,
            "full_doc": full_doc,
            "url": url,
            "page_index": page_index,
            "hyperlinks": hyperlinks
        }]
        res = self.client.insert(collection_name=self.collection, data=data)
        return res

    def load_data_from_generator(self, generator: Generator[dict, None, None], batch_size: int = 64):
        batch = []
        total = 0
        for kwargs in generator:
            batch.append(kwargs)
            if len(batch) >= batch_size:
                self._insert_batch(batch)
                total += len(batch)
                print(f"    Inserted {total} docs", flush=True)
                batch = []
        if batch:
            self._insert_batch(batch)
            total += len(batch)
            print(f"    Inserted {total} docs (done)", flush=True)

    def _insert_batch(self, batch: list[dict]):
        chunks = [item["chunk"] for item in batch]
        embeddings = self.embeder.encode_documents(chunks)
        data = []
        for item, emb in zip(batch, embeddings):
            data.append({
                "embeding": emb,
                "insurance_type": item["insurance_type"].value,
                "full_doc": item.get("full_doc", ""),
                "url": item.get("url", ""),
                "page_index": item.get("page_index", -1),
                "hyperlinks": item.get("hyperlinks", {}),
            })
        self.client.insert(collection_name=self.collection, data=data)

    def query_collection(self, insurance_type: InsuranceType, query: str, maximal_docs: int = 2):
        vectors = self.embeder.encode_queries([query])
        filter = f"insurance_type == {insurance_type.value}"
        output_fields = [
            "full_doc",
            "url",
            "page_index",
            "hyperlinks",
        ]
        res = self.client.search(collection_name=self.collection, data=vectors, filter=filter, output_fields=output_fields, limit=maximal_docs)[0]
        return res
