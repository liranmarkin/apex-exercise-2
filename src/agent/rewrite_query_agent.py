"""
Rewrite Query Agent.

This agent rewrites / refines a user's raw question into a concise, retrieval-optimized query.
It should be called after insurance type identification and before RAG retrieval.
"""

import sys
import re
import logging
from pathlib import Path
from langchain_nebius import ChatNebius

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import NEBIUS_API_KEY
from prompts import QUERY_REWRITE_PROMPT

# Module logger
logger = logging.getLogger(__name__)

# ============== QUERY REWRITE AGENT ==============

class QueryRewriteAgent:
    """Agent for rewriting user queries into retrieval-optimized forms."""
    
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        """
        Initialize the Query Rewrite Agent.
        
        Args:
            model_name: Nebius model name to use for query rewriting
        """
        self.model_name = model_name
        self.llm = ChatNebius(model=model_name, api_key=NEBIUS_API_KEY)
        logger.info(f"Initialized QueryRewriteAgent with model: {model_name}")
    
    def rewrite(self, user_input: str, insurance_type: str) -> str:
        """
        Rewrites the user's question into a retrieval-optimized query.

        Args:
            user_input: original user question
            insurance_type: identified insurance type name (e.g., 'CAR')

        Returns:
            A concise, focused string suitable for embedding & retrieval.
        """
        prompt = QUERY_REWRITE_PROMPT.format(
            insurance_type=insurance_type,
            user_input=user_input
        )
        try:
            llm_with_retries = self.llm.with_retry(
                stop_after_attempt=3,
                wait_exponential_jitter=True
            )

            response = llm_with_retries.invoke(
                prompt,
                temperature=0,           # Strictly factual
                max_tokens=150,          # Short search query only
                timeout=10,
            )

            text = response.content.strip()
            # Remove any <think> blocks or surrounding quotes
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            text = text.strip('"\'')
            # Fallback: if empty, return the original short form
            if not text:
                logger.warning("rewrite returned empty response; falling back to short user input")
                return (user_input[:200]).strip()
            logger.info("rewrote query: %s", text)
            return text
        except Exception:
            logger.exception("rewrite: LLM invocation failed")
            # Return a sensible fallback so upstream can continue
            return (user_input[:200]).strip()


# ============== BACKWARD COMPATIBILITY ==============
# Keep the old function signature for backward compatibility

def rewrite_query(user_input: str, insurance_type: str) -> str:
    """
    Rewrites the user's question into a retrieval-optimized query.

    Args:
        user_input: original user question
        insurance_type: identified insurance type name (e.g., 'CAR')

    Returns:
        A concise, focused string suitable for embedding & retrieval.
    """
    agent = QueryRewriteAgent()
    return agent.rewrite(user_input, insurance_type)
