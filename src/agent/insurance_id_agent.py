"""
Insurance Type Identification Agent.

This agent analyzes user queries and identifies which type of insurance they relate to.
"""

import sys
import logging
import time
import re
from pathlib import Path
from typing import Optional
from langchain_nebius import ChatNebius

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import InsuranceType, NEBIUS_API_KEY
from prompts import IDENTIFY_INSURANCE_TYPE_PROMPT

logger = logging.getLogger(__name__)


# ============== INSURANCE TYPE IDENTIFICATION AGENT ==============

class InsuranceTypeIdentificationAgent:
    """Agent for identifying insurance type from user queries."""
    
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        """
        Initialize the Insurance Type Identification Agent.
        
        Args:
            model_name: Nebius model name to use for classification
        """
        self.model_name = model_name
        self.llm = ChatNebius(model=model_name, api_key=NEBIUS_API_KEY)
        logger.info(f"Initialized InsuranceTypeIdentificationAgent with model: {model_name}")
    
    def identify(self, user_input: str) -> str:
        """
        Identifies the insurance type from the user query using LLM.
        
        Args:
            user_input: The user's question
            
        Returns:
            The identified insurance type as a string (enum name)
        """
        insurance_types = ', '.join([it.name for it in InsuranceType])
        identification_prompt = IDENTIFY_INSURANCE_TYPE_PROMPT.format(
            insurance_types=insurance_types,
            user_input=user_input
        )
        
        # invoke with retries
        insurance_type_str = ""
        llm_with_retries = self.llm.with_retry(
            stop_after_attempt=3,
            wait_exponential_jitter=True
        )

        try:
            resp = llm_with_retries.invoke(
                identification_prompt,
                temperature=0,           # Strictly factual
                max_tokens=800,          # Safety ceiling
                timeout=30,              # Don't let users hang forever
            )
        except Exception as exc:
            logger.exception("identify invoke failed after retries: %s", exc)
            return "CAR"  # Default fallback

        try:
            insurance_type_str = getattr(resp, "content", str(resp)).strip().upper()
        except Exception as exc:
            logger.exception("identify invoke failed: %s", exc)
            return "CAR"  # Default fallback
        
        # Validate the response
        try:
            InsuranceType[insurance_type_str]
            return insurance_type_str
        except KeyError:
            logger.error("Invalid insurance type returned: %s", insurance_type_str)
            return "CAR"  # Default fallback


# ============== BACKWARD COMPATIBILITY ==============
# Keep the old function signature for backward compatibility

def identify_insurance_type_from_query(user_input: str, max_retries: int = 2) -> str:
    """
    Identifies the insurance type from the user query using LLM.
    
    Args:
        user_input: The user's question
        
    Returns:
        The identified insurance type as a string
    """
    agent = InsuranceTypeIdentificationAgent()
    return agent.identify(user_input)