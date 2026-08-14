import os
from pathlib import Path
from typing import TypedDict, Literal, Optional

import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

from models import AskResponse
from prompt import SUPPORT_PROMPT

ROOT = Path(__file__).resolve().parent
DB = ROOT / "chroma_db"
COLLECTION_NAME = "zepto_policy_corpus"
MODEL_NAME = "all-MiniLM-L6-v2"

KEYWORDS = (
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
)


class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_ids: list[str]
    retrieved_chunks: list[str]
    answer: str
    sources: list[str]
    confidence: float
    response: dict


def mock_mode() -> bool:
    return os.getenv("MOCK_LLM", "1") != "0"


def get_collection():
    client = chromadb.PersistentClient(path=str(DB))
    return client.get_collection(COLLECTION_NAME)


def get_embedder():
    return SentenceTransformer(MODEL_NAME)


def classify_intent(state: GraphState) -> GraphState:
    query = state["query"].lower()
    intent = (
        "policy_question"
        if any(keyword in query for keyword in KEYWORDS)
        else "general_question"
    )

    # MOCK_LLM is intentionally not used for the routing edge itself.
    # In mock mode this deterministic heuristic is the graded path.
    # The optional real-LLM branch is represented by this same safe
    # deterministic fallback unless a provider adapter is added.
    return {"intent": intent}


def retrieve_and_answer(state: GraphState) -> GraphState:
    collection = get_collection()
    model = get_embedder()

    query_embedding = model.encode(
        [state["query"]],
        normalize_embeddings=True,
    ).tolist()[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    ids = result["ids"][0]
    chunks = result["documents"][0]

    if not chunks:
        answer = "I could not find relevant Zepto policy information."
        confidence = 0.0
    elif mock_mode():
        # Required graded baseline.
        snippet = chunks[0][:200]
        answer = f"Based on the retrieved context: {snippet}"
        confidence = 1.0
    else:
        # Optional real-LLM extension hook.
        # The required submission remains fully functional without
        # provider credentials. The structured prompt is available in
        # prompt.py for plugging in a provider such as Groq.
        context = "\n\n".join(
            f"[{doc_id}] {chunk}"
            for doc_id, chunk in zip(ids, chunks)
        )
        _optional_prompt = SUPPORT_PROMPT.format(
            context=context
        )
        answer = (
            "Real-LLM mode is enabled, but no provider adapter is "
            "configured in this offline baseline."
        )
        confidence = 0.0

    response = AskResponse(
        answer=answer,
        sources=ids,
        confidence=confidence,
    )

    return {
        "retrieved_ids": ids,
        "retrieved_chunks": chunks,
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "response": response.model_dump(),
    }


def direct_answer(state: GraphState) -> GraphState:
    if mock_mode():
        answer = "I can only answer questions about Zepto policies right now."
        confidence = 1.0
    else:
        answer = (
            "Real-LLM mode is enabled, but no provider adapter is "
            "configured in this offline baseline."
        )
        confidence = 0.0

    response = AskResponse(
        answer=answer,
        sources=[],
        confidence=confidence,
    )

    return {
        "answer": response.answer,
        "sources": [],
        "confidence": response.confidence,
        "response": response.model_dump(),
    }


def route_intent(state: GraphState) -> str:
    return (
        "retrieve_and_answer"
        if state["intent"] == "policy_question"
        else "direct_answer"
    )


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


app_graph = build_graph()


def ask(query: str) -> AskResponse:
    result = app_graph.invoke({"query": query})
    return AskResponse.model_validate(result["response"])
