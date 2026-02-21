"""
Response Generation Agent.

This agent generates answers using pre-computed RAG results.
"""

import sys
import re
import logging
from pathlib import Path
from langchain_nebius import ChatNebius

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import NEBIUS_API_KEY
from prompts import ANSWER_GENERATION_SYSTEM_PROMPT, ANSWER_GENERATION_PROMPT

# Module logger
logger = logging.getLogger(__name__)


# ============== ANSWER GENERATION AGENT ==============

class ResponseAgent:
    """Agent for generating answers using RAG results."""
    
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        """
        Initialize the Response Agent.
        
        Args:
            model_name: Nebius model name to use for answer generation
        """
        self.model_name = model_name
        self.llm = ChatNebius(model=model_name, api_key=NEBIUS_API_KEY)
        logger.info(f"Initialized ResponseAgent with model: {model_name}")

    def generate(self, user_input: str, insurance_type: str, rag_results: list) -> str:
        """
        Generates answer using pre-computed RAG results.
        
        Args:
            user_input: The user's question
            insurance_type: The identified insurance type
            rag_results: Pre-computed list of RAG results from retrieval
            
        Returns:
            The generated response with sources
        """
        # Format RAG results for the LLM
        context_section = ""
        if rag_results:
            context_section = "מסמכים רלוונטיים שהופקו:\n\n"
            for i, result in enumerate(rag_results, 1):
                doc = result.get("entity", {})
                url = doc.get("url", "N/A")
                page = doc.get("page_index", "N/A")
                content = doc.get("full_doc", "N/A")
                page_label = f", עמוד {page}" if page != "N/A" else ""
                context_section += f"מקור {i}:\n- כתובת: {url}{page_label}\n- תוכן: {content}...\n\n"
        else:
            context_section = "לא נמצאו מסמכים רלוונטיים."
        
        # Generate answer using LLM with RAG context
        answer_prompt = ANSWER_GENERATION_PROMPT.format(
            system_prompt=ANSWER_GENERATION_SYSTEM_PROMPT,
            insurance_type=insurance_type,
            user_input=user_input,
            context_section=context_section
        )
        
        try:
            llm_with_retries = self.llm.with_retry(
                stop_after_attempt=3,
                wait_exponential_jitter=True
            )

            response = llm_with_retries.invoke(
                answer_prompt,
                temperature=0,           # Strictly factual
                # max_tokens=1200,          # Safety ceiling
                timeout=30,              # Don't let users hang forever
            )

            # Remove reasoning/thinking tags if present
            clean_content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
            return clean_content
        except Exception:
            logger.exception("generate: LLM invocation failed")
            return "Error: failed to generate answer at this time. Please try again later."


# ============== BACKWARD COMPATIBILITY ==============
# Keep the old function signature for backward compatibility

def generate_answer_with_rag(user_input: str, insurance_type: str, rag_results: list) -> str:
    """
    Generates answer using pre-computed RAG results.
    
    Args:
        user_input: The user's question
        insurance_type: The identified insurance type
        rag_results: Pre-computed list of RAG results from retrieval
        
    Returns:
        The generated response with sources
    """
    agent = ResponseAgent()
    return agent.generate(user_input, insurance_type, rag_results)
