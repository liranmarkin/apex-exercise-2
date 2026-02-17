"""
Insurance Type Identification Agent.

This agent analyzes user queries and identifies which type of insurance they relate to.
"""

import sys
import re
from pathlib import Path
from langchain_nebius import ChatNebius

# Add src directory to Python path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import InsuranceType, NEBIUS_API_KEY


# Initialize LLM
llm = ChatNebius(model="meta-llama/Meta-Llama-3.1-8B-Instruct-fast", api_key=NEBIUS_API_KEY)


# ============== INSURANCE TYPE IDENTIFICATION ==============

def identify_insurance_type_from_query(user_input: str) -> str:
    """
    Identifies the insurance type from the user query using LLM.
    
    Args:
        user_input: The user's question
        
    Returns:
        The identified insurance type as a string
    """
    insurance_types = [it.name for it in InsuranceType]
    
    identification_prompt = f"""You are an insurance expert. Analyze the user query and identify which insurance type it relates to.

Available insurance types: {', '.join(insurance_types)}

User query: {user_input}

Return ONLY the insurance type name (e.g., CAR, LIFE, HEALTH). If unclear, return the most likely type."""
    
    response = llm.invoke(identification_prompt)
    insurance_type_str = response.content.strip().upper()
    
    # Validate the response
    try:
        InsuranceType[insurance_type_str]
        return insurance_type_str
    except KeyError:
        # If invalid, default to the closest match
        for it in InsuranceType:
            if it.name in insurance_type_str:
                return it.name
        return "CAR"  # Default fallback
