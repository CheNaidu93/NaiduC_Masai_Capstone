# Module 3 — Zepto Support Assistant

This module implements the required offline GenAI/RAG service for
the Zepto support-assistant assignment.

The graded baseline uses `MOCK_LLM` with its default value of `1`.
No LLM API key, signup, or external LLM network call is required.

## Architecture

```text
                 ┌──────────────────────┐
                 │  8 Zepto policy docs │
                 │       docs/*.txt     │
                 └──────────┬───────────┘
                            │
                       ingestion.py
                            │
                     chunk / document
                            │
                            ▼
                 SentenceTransformer
                 all-MiniLM-L6-v2
                            │
                            ▼
                    ChromaDB collection
                  zepto_policy_corpus
                            │
                            │
User ── POST /ask ──► FastAPI main.py
                            │
                            ▼
                    LangGraph StateGraph
                            │
                            ▼
                    classify_intent
                       /         \
                      /           \
          policy_question       general_question
                 │                    │
                 ▼                    ▼
       retrieve_and_answer       direct_answer
                 │                    │
          ChromaDB top-3          fixed mock response
                 │
                 ▼
         structured Pydantic JSON
          answer/sources/confidence
```

### Stage 1 — Ingestion

`docs/doc_01.txt` through `docs/doc_08.txt` contain the exact
assignment corpus. `ingest.py` reads all eight documents and stores
one document-sized chunk per file.

### Stage 2 — Embedding

`ingest.py` uses the local
`sentence-transformers` model `all-MiniLM-L6-v2`. Embeddings are
stored in a persistent ChromaDB collection named:

```text
zepto_policy_corpus
```

The ChromaDB database is stored under:

```text
chroma_db/
```

### Stage 3 — Retrieval

The `retrieve_and_answer` LangGraph node embeds the incoming query
and asks ChromaDB for the top 3 most similar documents using cosine
similarity.

Retrieval runs in both mock and optional real-LLM modes.

### Stage 4 — Generation

For the graded default `MOCK_LLM=1` mode, no LLM call is made.

`retrieve_and_answer` creates:

```text
Based on the retrieved context: <top chunk snippet>
```

`direct_answer` returns:

```text
I can only answer questions about Zepto policies right now.
```

The final result is validated with the Pydantic `AskResponse` schema:

```json
{
  "answer": "string",
  "sources": ["document IDs"],
  "confidence": 0.0
}
```

## MOCK_LLM behavior

The environment variable is controlled by:

```text
MOCK_LLM
```

Default:

```text
MOCK_LLM=1
```

or unset.

The intent classifier uses the assignment's required keyword
heuristic. These keywords trigger a `policy_question`:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

Everything else is routed to `general_question`.

The assignment requires this deterministic path to work without any
LLM call.

`MOCK_LLM=0` is an optional extension. The code contains the structured
prompt template and a provider-independent hook, but the required
submission does not depend on a provider API.

## Structured prompt

`prompt.py` contains the required role-context-task-format-length
skeleton.

It also contains:

- an explicit negative constraint: do not answer using information
  outside the provided context
- a few-shot example
- JSON output instructions
- source and confidence requirements

The prompt is intended for the optional real-LLM extension.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Build the vector index

Run:

```bash
python ingest.py
```

Expected final message:

```text
Indexed 8 documents.
```

This creates:

```text
chroma_db/
```

## Run the automated mock-mode tests

```bash
python test_mock.py
```

The test demonstrates:

1. A policy query routes to retrieval.
2. A general query routes directly.
3. The policy response contains sources.
4. The general response has an empty sources list.
5. Mock confidence is deterministic.

## Run FastAPI locally

Make sure `MOCK_LLM` is unset or set to `1`:

Windows CMD:

```cmd
set MOCK_LLM=1
```

PowerShell:

```powershell
$env:MOCK_LLM="1"
```

Linux/macOS:

```bash
export MOCK_LLM=1
```

Then:

```bash
uvicorn main:app --reload
```

The service runs at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Example call 1 — policy question

Request:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Is delivery free for orders over INR 149?\"}"
```

Representative mock response:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.",
  "sources": [
    "doc_01",
    "doc_03",
    "doc_08"
  ],
  "confidence": 1.0
}
```

The exact order of the lower-ranked sources can depend on the local
embedding/index version. The important acceptance criterion is that
the top retrieved document matches the question, namely `doc_01`.

## Example call 2 — general question

Request:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What is the capital of France?\"}"
```

Response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

No retrieval is needed for this query.

## Test another policy question

For returns:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"How long do I have to report a damaged grocery item?\"}"
```

The relevant source should include `doc_02` and/or `doc_06`.

For membership:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"How much does Zepto Pass cost?\"}"
```

The relevant source should be `doc_03`.

For support hours:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What are the support hours?\"}"
```

The relevant source should be `doc_08`.

## Docker

Build:

```bash
docker build -t zepto-support-assistant .
```

Run:

```bash
docker run --rm -p 7860:7860 zepto-support-assistant
```

Then test:

```bash
curl -X POST "http://127.0.0.1:7860/ask" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Can I cancel my order?\"}"
```

The Docker image builds the ChromaDB index during the image build,
so the running container already has the eight embedded policy
documents.

## Files

```text
support_assistant/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── ingest.py
├── graph.py
├── prompt.py
├── models.py
├── main.py
├── test_mock.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Assignment checklist

- [x] 8 policy documents included
- [x] Local `all-MiniLM-L6-v2` embeddings
- [x] ChromaDB persistent vector collection
- [x] Structured role/context/task/format/length prompt
- [x] Negative prompt constraint
- [x] Few-shot prompt example
- [x] LangGraph `StateGraph`
- [x] TypedDict graph state
- [x] `classify_intent` node
- [x] `retrieve_and_answer` node
- [x] `direct_answer` node
- [x] Conditional routing
- [x] Deterministic MOCK_LLM baseline
- [x] Top-3 ChromaDB retrieval
- [x] Pydantic output validation
- [x] `answer`, `sources`, `confidence`
- [x] FastAPI `POST /ask`
- [x] Two example requests
- [x] Dockerfile
- [x] Ingestion → embedding → retrieval → generation architecture
- [x] Explanation of MOCK_LLM branching

## Important grading note

The assignment's required graded baseline is the offline mock mode.
Do not make the submission depend on a real LLM API key.

The optional real-LLM and Hugging Face deployment extensions are not
necessary for full marks.
