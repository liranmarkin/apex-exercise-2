import time
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Optional
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.harel_agent import HarelAgent

agent = HarelAgent()

logger = logging.getLogger("api.completions_api")

app = FastAPI(title="Completions-Compatible REST API")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mock-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

class Source(BaseModel):
    link: str
    page: int

class Choice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = None
    sources: List[Source] = []

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: float
    model: str
    choices: List[Choice]

# ----------------------
# Endpoints
# ----------------------

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def question_endpoint(request: ChatCompletionRequest):
    response = await process_completions_request(request)
    return response

@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def question_endpoint(request: ChatCompletionRequest):
    response = await process_completions_request(request)
    return response

async def process_completions_request(request: ChatCompletionRequest) -> ChatCompletionResponse:
    logger.info(f"Processing chat completion request: messages={len(request.messages)}")

    # Build a simple prompt from incoming messages (prefer last user message)
    if len(request.messages) == 1:
        user_questions = request.messages[0].content
        try:
            answer_text, stats = await asyncio.to_thread(agent.chat, user_questions, generate_stats=True)
            logger.info(f"Generated answer (len={len(answer_text)} chars)")
        except Exception as e:
            logger.exception("Failed to generate answer via HarelAgent")
            answer_text = "Error: failed to generate answer at this time."
    else:
        answer_text = "Error: multiple messages not supported"
    
    #  Example: [Source(link="https://www.harel-group.co.il/insurance/car/policies/motorcycle", page=4")
    retrieved_sources = stats["intermediate"]["retrieved_sources"]
    source_list = [Source(link=src["url"], page=src.get("page_index", -1)) for src in retrieved_sources]
    return ChatCompletionResponse(
        id="question-1",
        object="chat.completion",
        created=time.time(),
        model="mock-model",
        choices=[
            Choice(
                index=0,
                text=answer_text,
                sources=source_list,
                finish_reason="stop"
            )
        ]
    )

# ----------------------
# Run with:
# uvicorn myapp:app --reload
# ----------------------