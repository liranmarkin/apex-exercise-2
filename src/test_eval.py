import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Setup Nebius (Judge) & Local Embeddings
from constants import NEBIUS_API_KEY

# Wrap Nebius Chat
nebius_llm = ChatOpenAI(
    model="Qwen/Qwen3-32B", # Or Qwen2.5-72B-Instruct
    api_key=NEBIUS_API_KEY,
    base_url="https://api.studio.nebius.ai/v1"
)
evaluator_llm = LangchainLLMWrapper(nebius_llm)

# Wrap Local Embeddings (Free, runs on CPU/GPU)
# pip install langchain-huggingface sentence-transformers
local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
evaluator_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

# 2. Dataset with NEW Naming Conventions (Ragas 0.2+)
# user_input = Question
# response = Bot's Answer
# retrieved_contexts = Milvus Snippets (Must be a list of lists)
# reference = Ground Truth (Correct Answer)
data_samples = {
    "user_input": [
        "What is the waiting period for dental insurance at Harel?",
        "Does Harel cover international travel accidents?"
    ],
    "response": [
        "There is a 3-month waiting period for dental care.",
        "Yes, it covers worldwide accidents."
    ],
    "retrieved_contexts": [
        ["Policy Section 4.2: Dental care requires a 3-month qualification period."],
        ["Travel Appendix: Worldwide accident coverage is included."]
    ],
    "reference": [
        "A 3-month waiting period applies to dental insurance.",
        "Harel provides worldwide coverage for accidents."
    ]
}

dataset = Dataset.from_dict(data_samples)

# 3. Import and Run all 4 Metrics
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_precision, 
    context_recall
)

results = evaluate(
    dataset,
    metrics=[
        faithfulness, 
        answer_relevancy, 
        context_precision, 
        context_recall
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

# 4. Display Final DataFrame
df = results.to_pandas()
print("\n--- Full RAGAS Evaluation Results ---")
# Note: output columns match the input names + metric names
print(df[['user_input', 'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']])