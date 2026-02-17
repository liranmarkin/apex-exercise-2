"""
Harel Insurance Customer Support Chatbot.

This module orchestrates two agents:
1. insurance_id_agent: Identifies the insurance type from user input
2. response_agent: Generates answers using RAG with the identified insurance type
"""

import sys
import time
from pathlib import Path

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from insurance_id_agent import identify_insurance_type_from_query
from response_agent import generate_answer_with_rag


# ============== MAIN CHAT FUNCTION ==============

def chat(user_input: str) -> str:
    """
    Main chat function that orchestrates the two-agent system.
    
    This function:
    1. Uses Agent 1 to identify the insurance type from the user query
    2. Uses Agent 2 to generate an answer using RAG with the identified type
    
    Args:
        user_input: The user's question in Hebrew or English
        
    Returns:
        A response string with the answer and sources
    """
    
    # Agent 1: Identify Insurance Type
    print("\n[Agent 1] Starting insurance type identification...")
    start_time = time.time()
    insurance_type = identify_insurance_type_from_query(user_input)
    agent1_time = time.time() - start_time
    print(f"[Agent 1] Identified Insurance Type: {insurance_type}")
    print(f"[Agent 1] Execution Time: {agent1_time:.2f} seconds")
    
    # Agent 2: Generate Answer with RAG
    print("\n[Agent 2] Starting answer generation with RAG...")
    start_time = time.time()
    final_answer = generate_answer_with_rag(user_input, insurance_type)
    agent2_time = time.time() - start_time
    print(f"[Agent 2] Execution Time: {agent2_time:.2f} seconds")
    
    # Total time
    total_time = agent1_time + agent2_time
    print(f"\n[Total] Combined Execution Time: {total_time:.2f} seconds")
    
    return final_answer


def main():
    """Test the chat function with a sample query."""
    sample_query = "אני מבוטח בהראל בפוליסה לביטוח צ.מ.ה. השתמשתי במנוע של כלי מבוטח כדי להפעיל את המצבר של רכב אחר של העסק שלי, ונגרם נזק לכלי המבוטח. האם אני זכאי לכיסוי על הנזק במקרה כזה?"
    
    print("User Query:")
    print(sample_query)
    print("\n" + "="*80 + "\n")
    
    response = chat(sample_query)
    
    print("Chatbot Response:")
    print(response)


if __name__ == "__main__":
    main()
