from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from guardrailed_bot import GuardrailedTNAgriBot

bot_instance: GuardrailedTNAgriBot | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot_instance
    print("Loading Knowledge Graph and NeMo Guardrails...")
    bot_instance = GuardrailedTNAgriBot()
    print("TN Agriculture GraphRAG API is ready!")
    yield
    bot_instance = None


app = FastAPI(
    title="Tamil Nadu Agriculture Schemes GraphRAG API",
    description="Guardrailed GraphRAG assistant powered by LangChain, OpenAI, NetworkX, and NVIDIA NeMo Guardrails.",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, description="User question about TN agriculture schemes")


class ChatResponse(BaseModel):
    question: str
    answer: str
    retrieval_context: List[str]
    guardrail_blocked: bool


@app.get("/health")
async def health_check():
    return {"status": "ok", "bot_loaded": bot_instance is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    if bot_instance is None:
        raise HTTPException(status_code=503, detail="Bot engine is still initializing.")

    result = await bot_instance.chat(payload.question)
    blocked = (
        len(result["retrieval_context"]) == 0
        and "only assist with Tamil Nadu Agriculture" in result["answer"]
    )
    return ChatResponse(
        question=result["question"],
        answer=result["answer"],
        retrieval_context=result["retrieval_context"],
        guardrail_blocked=blocked,
    )