"""
Response Generation Agent.

This agent generates answers using RAG with the identified insurance type.
"""

import sys
import re
from pathlib import Path
from langchain_nebius import ChatNebius

# Add src directory to Python path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import InsuranceType, NEBIUS_API_KEY
from rag.mock_rag import MockRAG


# Initialize RAG and LLM
rag = MockRAG()
llm = ChatNebius(model="openai/gpt-oss-120b", api_key=NEBIUS_API_KEY)


# ============== ANSWER GENERATION ==============

def generate_answer_with_rag(user_input: str, insurance_type: str) -> str:
    """
    Generates answer using RAG with the identified insurance type.
    
    Args:
        user_input: The user's question
        insurance_type: The identified insurance type
        
    Returns:
        The generated response with sources
    """
    try:
        insurance_type_enum = InsuranceType[insurance_type]
    except KeyError:
        return f"Error: Invalid insurance type '{insurance_type}'"
    
    # Query RAG
    rag_results = rag.query_collection(insurance_type_enum, user_input, maximal_docs=2)
    
    # Format RAG results for the LLM
    sources_text = ""
    if rag_results:
        sources_text = "Retrieved relevant documents:\n\n"
        for i, result in enumerate(rag_results, 1):
            doc = result.get("entity", {})
            url = doc.get("url", "N/A")
            page = doc.get("page_index", "N/A")
            content = doc.get("full_doc", "N/A")
            sources_text += f"Source {i}:\n- URL: {url}\n- Page: {page}\n- Content: {content[:500]}...\n\n"
    
    # Generate answer using LLM with RAG context
    answer_prompt = f"""You are a helpful Harel Insurance customer support agent.

User Query: {user_input}

Insurance Type: {insurance_type}

{sources_text}

Based on the retrieved documents above, provide a helpful and accurate answer to the user's question.
At the end, list the sources you used with their URLs and page numbers in this format:
Sources:
- [Source Title](URL) - Page [number]"""
    
    response = llm.invoke(answer_prompt)
    # Remove reasoning/thinking tags if present
    clean_content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
    return clean_content
