from pymilvus import MilvusClient, DataType, model

import constants
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



class RAG:
    def __init__(self, reset_collection: bool = True):
        self.embeder = self._get_embeder()
        self.collection = constants.COLLECTION_NAME
        self.client = self._get_db_client(DB_PATH)
        self.schema = self._get_schema()

        # Initialize LLM
        self.llm = ChatOpenAI(api_key=constants.OPENAI_API_KEY, model="gpt-4o", temperature=0)

        
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

    def classify_query(self, query: str) -> str:
        prompt = PromptTemplate.from_template(
            "Classify the following question into one of these insurance domains: {domains}.\n"
            "Question: {question}\n"
            "Return ONLY the domain name."
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            domain = chain.invoke({"domains": ", ".join(constants.INSURANCE_TYPES), "question": query}).strip()
        except Exception as e:
            print(f"Classification failed: {e}")
            return "General"
        
        # Cleanup and validation
        for d in constants.INSURANCE_TYPES:
            if d.lower() == domain.lower():
                return d
        # Keyword fallback
        for d in constants.INSURANCE_TYPES:
            if d.lower() in query.lower():
                return d
        return "General"

    def answer_question(self, query: str) -> str:
        domain = self.classify_query(query)
        print(f"Classified domain: {domain}")
        
        docs = self.query_collection(domain, query)
        
        # MilvusClient search result handling
        context_parts = []
        for d in docs:
            text = d.get('entity', {}).get('document')
            if not text:
                text = d.get('document')
            if text:
                context_parts.append(f"Document: {text}")
        
        context = "\n\n".join(context_parts)
        
        if not context:
            return "I cannot answer this based on the provided documents (No relevant documents found)."

        prompt = PromptTemplate.from_template(
            "You are a helpful customer support agent for Harel Insurance.\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Rules:\n"
            "1. Answer ONLY based on the context.\n"
            "2. If the answer is not in the context, say 'I cannot answer this based on the provided documents.'\n"
            "3. Cite your sources for every fact using the format [Document Name, Page/Section].\n"
            "Answer:"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})

