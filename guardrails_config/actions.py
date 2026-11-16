from typing import Dict, Any
from nemoguardrails.actions import action
from graph_rag import TNAGriGraphRAG

# Initialize the GraphRAG engine once when actions load
rag_engine = TNAGriGraphRAG()

# Store the latest RAG trace (answer + retrieval_context) per request
LAST_RAG_TRACE: Dict[str, Any] = {"retrieval_context": []}


@action(name="query_tn_agri_kg")
async def query_tn_agri_kg(question: str) -> str:
    result = rag_engine.ask(question)
    LAST_RAG_TRACE["retrieval_context"] = result.get("retrieval_context", [])
    return result["answer"]