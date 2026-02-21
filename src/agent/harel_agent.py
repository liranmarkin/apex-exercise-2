"""
Harel Insurance Customer Support Chatbot.

This module orchestrates three agents:
1. insurance_id_agent: Identifies the insurance type from user input
2. rewrite_query_agent: Rewrites the query for retrieval
3. response_agent: Generates answers using RAG results
"""

import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Module logger
logger = logging.getLogger(__name__)

# Add src and agent directories to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))  # agent directory
# sys.path.insert(0, str(Path(__file__).parent.parent))  # src directory

from insurance_id_agent import InsuranceTypeIdentificationAgent
from rewrite_query_agent import QueryRewriteAgent
from response_agent import ResponseAgent
from rag.mock_rag import MockRAG
from rag.rag import RAG
from constants import InsuranceType


# ============== HAREL AGENT ORCHESTRATOR ==============

class HarelAgent:
    """Main orchestrator agent for Harel Insurance customer support chatbot."""
    
    FAST_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-fast"
    MEDIUM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-fast"
    STRONG_MODEL = "openai/gpt-oss-120b"

    def __init__(self):
        self.identification_agent = InsuranceTypeIdentificationAgent(model_name=self.FAST_MODEL)
        self.rewrite_agent = QueryRewriteAgent(model_name=self.MEDIUM_MODEL)
        self.response_agent = ResponseAgent(model_name=self.STRONG_MODEL)
        self.rag = RAG(reset_collection=False)
        # Warm up Milvus index to avoid cold-start latency on the first real query
        _warmup_vec = self.rag.embeder.encode_queries(["warmup"])[0]
        self.rag.client.search(collection_name=self.rag.collection, data=[_warmup_vec], limit=1)

        logger.info("Initialized HarelAgent with all subagents")
    
    def _deduplicate_and_rank_documents(self, hits, top_k=5):
        """
        Deduplicate documents by (url, page_index), keeping the best distance per key.
        Score, filter, and rank — take top k.
        
        Args:
            hits: List of document hits from retrieval
            top_k: Maximum number of documents to return
            
        Returns:
            List of up to 5 deduplicated and ranked documents
        """
        # Deduplicate by (url, page_index), keeping the best distance per key
        seen = {}
        for hit in (hits or []):
            url = hit["entity"].get("url", "")
            page = hit["entity"].get("page_index", -1)
            key = (url, page)
            dist = hit.get("distance", 1.0)  # Default to 1.0 for mock data
            if key not in seen or dist > seen[key].get("distance", 0):
                seen[key] = {
                    "entity": hit["entity"],
                    "distance": dist,
                }

        # Score, filter, and rank — take top 5
        MIN_DISTANCE = 0.45  # reject low-similarity results
        FAQ_BOOST = 1.15
        candidates = []
        for key, hit in seen.items():
            dist = hit["distance"]
            if dist < MIN_DISTANCE:
                continue
            source_type = hit["entity"].get("source_type", 2)
            score = dist * FAQ_BOOST if source_type == 1 else dist  # FAQ boost
            candidates.append((score, key, hit))
        
        # Sort by score (descending) and take top k
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:top_k]

        # Reconstruct the hit format for return
        result = []
        for score, key, hit in top:
            result.append({
                "entity": hit["entity"],
                "distance": hit["distance"],
            })
        
        return result
    
    def chat(self, user_input: str, generate_stats: bool = False):
        """
        Main chat function that orchestrates the three-agent system.
        
        This function:
        1. Uses Agent 1 to identify the insurance type from the user query
        2. Uses Agent 2 to rewrite the query for retrieval
        3. Performs RAG retrieval with the rewritten query
        4. Uses Agent 3 to generate an answer using RAG results
        
        Args:
            user_input: The user's question in Hebrew or English
            generate_stats: If True, returns (answer, stats_dict); if False, returns just the answer
            
        Returns:
            If generate_stats=False: answer string
            If generate_stats=True: tuple of (answer string, stats_dict with timings and intermediate results)
        """
        
        pipeline_start = time.time()

        # Agent 1 + Agent 2: Run in parallel (rewrite doesn't need insurance type for good results)
        logger.info("Starting insurance type identification + query rewrite in parallel...")
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_id = executor.submit(self.identification_agent.identify, user_input)
            future_rewrite = executor.submit(self.rewrite_agent.rewrite, user_input, "GENERAL")
            insurance_type = future_id.result()
            rewritten_query = future_rewrite.result()
        parallel_time = time.time() - start_time
        agent1_time = parallel_time
        rewrite_time = parallel_time
        logger.info(f"Agent 1+2 completed in parallel - Type: {insurance_type}, Query: {rewritten_query} (Time: {parallel_time:.2f}s)")

        # RAG Retrieval
        logger.info("Starting document retrieval...")
        start_time = time.time()
        try:
            insurance_type_enum = InsuranceType[insurance_type]
            raw_results = self.rag.query_collection(insurance_type_enum, rewritten_query, maximal_docs=15)
            # Deduplicate and rank documents
            rag_results = self._deduplicate_and_rank_documents(raw_results)
            logger.info(f"RAG retrieval after deduplication:\n{rag_results}\n")
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            rag_results = []
        rag_time = time.time() - start_time
        logger.info(f"RAG retrieval completed - Retrieved {len(rag_results)} documents after deduplication (Time: {rag_time:.2f}s)")

        # Agent 3: Generate Answer with RAG using rewritten query and pre-retrieved results
        logger.info("Starting answer generation with RAG...")
        start_time = time.time()
        final_answer = self.response_agent.generate(rewritten_query, insurance_type, rag_results)
        agent3_time = time.time() - start_time
        logger.info(f"Agent 3 completed - Generated answer (Time: {agent3_time:.2f}s)")
        logger.debug(f"Agent 3 response: {final_answer}...") if len(final_answer) > 200 else logger.debug(f"Agent 3 response: {final_answer}")
        
        # Total time
        total_time = agent1_time + rewrite_time + rag_time + agent3_time
        logger.info(f"Combined Execution Time: {total_time:.2f}s - Agent 1: {agent1_time:.2f}s, Agent 2: {rewrite_time:.2f}s, RAG: {rag_time:.2f}s, Agent 3: {agent3_time:.2f}s")
        
        # Return with or without stats
        if generate_stats:
            stats = {
                "identification_s": round(agent1_time, 3),
                "rewrite_s": round(rewrite_time, 3),
                "retrieval_s": round(rag_time, 3),
                "llm_answer_s": round(agent3_time, 3),
                "total_s": round(total_time, 3),
                "intermediate": {
                    "identified_type": insurance_type,
                    "rewritten_query": rewritten_query,
                    "retrieved_docs_count": len(rag_results),
                    "retrieved_docs": [r.get("entity", {}).get("full_doc", "") for r in rag_results] if rag_results else [],
                    "retrieved_sources": [
                        {
                            "url": r.get("entity", {}).get("url", ""),
                            "page_index": r.get("entity", {}).get("page_index", -1),
                        }
                        for r in rag_results
                    ] if rag_results else [],
                }
            }
            # Log stats information
            logger.info(f"Stats - Identification: {stats['identification_s']:.3f}s, Rewrite: {stats['rewrite_s']:.3f}s, Retrieval: {stats['retrieval_s']:.3f}s, LLM: {stats['llm_answer_s']:.3f}s, Total: {stats['total_s']:.3f}s")
            logger.info(f"Stats - Insurance Type: {stats['intermediate']['identified_type']}, Docs Retrieved: {stats['intermediate']['retrieved_docs_count']}")
            logger.info(f"Stats - Rewritten Query: {stats['intermediate']['rewritten_query']}")
            return final_answer, stats
        else:
            return final_answer




def main():
    """Test the Harel Agent with a sample query."""
    # Configure logging to display info and above
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    sample_query = "אני מבוטח בהראל בפוליסה לביטוח צ.מ.ה. השתמשתי במנוע של כלי מבוטח כדי להפעיל את המצבר של רכב אחר של העסק שלי, ונגרם נזק לכלי המבוטח. האם אני זכאי לכיסוי על הנזק במקרה כזה?"
    
    logger.info(f"User Query: {sample_query}")
    
    # Create and run the Harel Agent
    agent = HarelAgent()
    response = agent.chat(sample_query)
    
    logger.info(f"Chatbot Response: {response}")


if __name__ == "__main__":
    main()