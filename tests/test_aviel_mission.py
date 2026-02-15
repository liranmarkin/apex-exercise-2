import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from rag.rag import RAG
import constants

def main():
    # Load reference questions
    json_path = os.path.join(os.path.dirname(__file__), "../reference-questions.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {json_path}")
        return

    print("Initializing RAG system...")
    try:
        rag = RAG(reset_collection=False)
    except Exception as e:
        print(f"Failed to initialize RAG: {e}")
        return

    print("\nStarting Verification...")
    
    total_tests = 0
    passed_tests = 0
    
    # Iterate over domains
    for domain, questions in data.items():
        print(f"\n--- Testing Domain: {domain} ---")
        # Take first 2 questions from each domain to save time/cost
        for i, q_item in enumerate(questions[:2]):
            total_tests += 1
            question = q_item["שאלה"]
            expected_answer = q_item["תשובה"]
            
            print(f"\nQ{i+1}: {question}")
            
            # 1. Test Classification
            inferred_domain = rag.classify_query(question)
            print(f"  Inferred Domain: {inferred_domain}")
            
            # Check if inferred domain matches mapping (Hebrew to English)
            # define mapping or just print it.
            # INSURANCE_TYPES are English. Keys in JSON are Hebrew.
            # We can't strictly assert equality without a map, but we can see if it makes sense.
            
            # 2. Test Answering
            try:
                # We use the method which calls classify internally, but here we can see the domain.
                answer = rag.answer_question(question)
                print(f"  Answer: {answer}")
                print(f"  Expected (approx): {expected_answer}")
                
                if "I cannot answer" not in answer:
                    passed_tests += 1
                else:
                    print("  [WARNING] RAG refused to answer.")

            except Exception as e:
                print(f"  [ERROR] Failed to answer: {e}")

    print(f"\nVerification Complete. Attempted {total_tests} questions.")
    print(f"Note: 'Passed' just means it generated an answer, not necessarily correct.")

if __name__ == "__main__":
    main()
