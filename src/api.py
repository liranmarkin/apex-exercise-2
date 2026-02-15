from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from rag.rag import RAG
import uvicorn
import sys
import os
import constants
from langchain_openai import ChatOpenAI

# Add src to path if needed (though usually handled by execution context)
sys.path.append(os.path.dirname(__file__))

app = FastAPI(title="Harel Insurance Chatbot")

# Initialize RAG system
# We use reset_collection=False because we assume data is already ingested
# or will be ingested by another process.
try:
    rag_system = RAG(reset_collection=False)
except Exception as e:
    print(f"Warning: Failed to initialize RAG system (Database might be offline): {e}")
    rag_system = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not rag_system:
         raise HTTPException(status_code=503, detail="RAG system is not initialized")
    
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        answer = rag_system.answer_question(request.query)
        
        return QueryResponse(answer=answer, sources=[])
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simple-chatgpt", response_model=QueryResponse)
async def simple_chatgpt(request: QueryRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        llm = ChatOpenAI(api_key=constants.OPENAI_API_KEY, model="gpt-4o", temperature=0.7)
        response = llm.invoke(request.query)
        
        return QueryResponse(answer=response.content, sources=[])
    except Exception as e:
        print(f"Error calling ChatGPT: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    pass


"""
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d "{\"query\": \"How are you?\"}"
"""
