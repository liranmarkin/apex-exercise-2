from constants import InsuranceType
from rag.rag import RAG

LOAD_DATA = False


if LOAD_DATA:
    def sample_generator():
        data = [
            ("Artificial intelligence was founded as an academic discipline in 1956.", InsuranceType.LIFE, "FULLDOC_1", "URL_1"),
            ("Alan Turing was the first person to conduct substantial research in AI.", InsuranceType.LIFE, "FULLDOC_2", "URL_2", 2),
            ("Born in Maida Vale, London, Turing was raised in southern England.", InsuranceType.LIFE, "FULLDOC_3", "URL_3", 3, {"hyperlink_text_1": "hyperlink_1", "hyperlink_text_2": "hyperlink_2"}),
        ]
        for record in data:
            kwargs = dict()
            kwargs["chunk"] = record[0] # Type: str
            kwargs["insurance_type"] = record[1] # Type: InsuranceType
            kwargs["full_doc"] = record[2] # Type: str
            kwargs["url"] = record[3] # Type: str
            # These are optional
            if len(record) > 4:
                kwargs["page_index"] = record[4] # Type: int
            if len(record) > 5:
                kwargs["hyperlinks"] = record[5] # Type: dict[str, str]
           
            yield kwargs

    rag = RAG(reset_collection=True)
    rag.load_data_from_generator(sample_generator())

else:
    rag = RAG()

print(rag.query_collection(InsuranceType.LIFE, "Who is Alan Turing?"))
print()
print(rag.query_collection(InsuranceType.LIFE, "Artifitial Intelegence was founded as what?"))
print()
print(rag.query_collection(InsuranceType.TRAVEL, "Who is Alan Turing?", maximal_docs=1))

# query_collection output format:
# {
#     'id': int, # randomly  generated
#     'distance': float, # 1 means close
#     'entity': { # This is the result data
#         'full_doc': str, # Text to insert into LLM context
#         'url': str, # Url that the file can be found at
#         'page_index': int, # relevant page for the data, this is an optional result, and if the data is missing it will return -1
#         'hyperlinks': dict[str, str], # If the result has hyperlinks, this dict will hold hyperlink_text: hyperlink_url, deafult is an empty dict
#     }
# }
