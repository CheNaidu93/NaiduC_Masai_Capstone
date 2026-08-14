from fastapi import FastAPI
from models import AskRequest, AskResponse
from graph import ask

app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0.0",
    description="Offline RAG support assistant using LangGraph, ChromaDB and local embeddings.",
)


@app.get("/")
def root():
    return {
        "service": "Zepto Support Assistant",
        "mock_llm": "enabled by default",
        "endpoint": "POST /ask",
    }


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    return ask(request.query)
