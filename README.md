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
# Data Pipeline — Zepto Capstone

## Overview

This module implements an end-to-end data engineering pipeline for catalog-style product data.

The pipeline performs the following steps:

1. Scrapes book data from Books to Scrape.
2. Cleans and validates the scraped fields.
3. Converts GBP prices to INR using the required fixed project rate.
4. Stores the cleaned data in a normalized SQLite database.
5. Executes SQL queries demonstrating filtering, sorting, limiting, distinct values, range filtering and joins.
6. Reads SQL results into pandas DataFrames.
7. Reproduces the SQL JOIN using `pandas.merge()` and validates that both approaches produce equivalent results.

The project uses the same raw-to-relational workflow described in the capstone brief: raw scraped data is transformed into a clean relational store for downstream analytics.

---

## Project Structure

```text
data_pipeline/
│
├── scraper.py
├── database.py
├── queries.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── sql_outputs.txt
└── books.db
```

---

## Data Source

The data source is:

`http://books.toscrape.com/`

Books to Scrape is a public website intended for scraping practice.

No login, API key or paid service is required.

---

## Scraping Scope

The pipeline dynamically discovers book categories and scrapes complete category pagination until at least:

* 60 books
* 3 different categories

have been collected.

This ensures that the final dataset satisfies the assignment requirement while avoiding hard-coded product data.

---

## Fields Collected

The scraper initially collects:

| Field          | Description                                       |
| -------------- | ------------------------------------------------- |
| `title`        | Book title                                        |
| `price`        | Original price text in GBP                        |
| `star_rating`  | Rating text such as One, Two, Three, Four or Five |
| `availability` | Original availability text                        |
| `category`     | Book category                                     |

The cleaning stage creates the required normalized fields:

| Field       | Type    | Description                  |
| ----------- | ------- | ---------------------------- |
| `price_gbp` | float   | Numeric GBP price            |
| `price_inr` | float   | Converted INR price          |
| `rating`    | integer | Rating from 1 to 5           |
| `in_stock`  | boolean | Whether the book is in stock |
| `category`  | string  | Category name                |

---

## Currency Conversion

The required fixed project baseline is:

**1 GBP = 105.50 INR**

This is an artificial project-defined conversion rate.

No live currency API is required or used.

The INR price is calculated as:

```text
price_inr = price_gbp × 105.50
```

The result is rounded to two decimal places.

---

## Cleaning Decisions

### Price

The currency symbol and other non-numeric characters are removed before conversion to `float`.

Example:

```text
£51.77 → 51.77
```

If a price cannot be parsed, the numeric value is represented as missing initially and median imputation is applied.

### Rating

The website provides textual ratings:

```text
One
Two
Three
Four
Five
```

These are converted using:

```text
One   → 1
Two   → 2
Three → 3
Four  → 4
Five  → 5
```

If a numeric rating cannot be parsed, the median rating is used.

### Availability

Availability is converted to a Boolean value.

If the availability text contains `In stock`, the resulting value is:

```text
True
```

Otherwise:

```text
False
```

### Missing essential fields

Rows without an identifiable title or category are dropped because these fields are required to uniquely identify and classify a book.

For numeric fields, median imputation is used instead of dropping otherwise usable rows.

---

## Database Design

SQLite is used as the relational database.

The database contains two normalized tables.

### categories

```sql
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
);
```

### books

```sql
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,

    FOREIGN KEY (category_id)
        REFERENCES categories(category_id),

    CHECK (rating BETWEEN 1 AND 5),

    CHECK (in_stock IN (0, 1))
);
```

The relationship is:

```text
categories
    │
    │ category_id
    │
    └──────────< books.category_id
```

This avoids storing the category name repeatedly in every database row.

---

## SQL Queries

The pipeline executes six SQL queries.

They collectively demonstrate:

* `SELECT`
* `WHERE`
* `ORDER BY`
* `LIMIT`
* `DISTINCT`
* `BETWEEN`
* `IN`
* `JOIN`

The executed outputs are stored in:

```text
sql_outputs.txt
```

The JOIN query combines the `books` and `categories` tables using:

```sql
JOIN categories c
    ON b.category_id = c.category_id
```

---

## Pandas Validation

At least two SQL query results are loaded using:

```python
pd.read_sql()
```

The JOIN result is independently reproduced using:

```python
pd.merge()
```

The SQL and pandas results are sorted using the same ordering columns and compared using:

```python
DataFrame.equals()
```

The pipeline prints whether both results are equivalent.

---

## Installation

From the `data_pipeline` directory:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Run:

```bash
python run_pipeline.py
```

The pipeline will:

1. Scrape the website.
2. Clean the data.
3. Convert GBP to INR.
4. Create `books.db`.
5. Populate the normalized tables.
6. Execute the SQL queries.
7. Save query outputs to `sql_outputs.txt`.
8. Reproduce the JOIN using pandas.
9. Compare the SQL JOIN and pandas JOIN results.

The pipeline is designed to run from scratch without manual copy-pasting of data.

---

## Reproducibility

The SQLite tables are recreated each time the pipeline runs.

Therefore:

```bash
python run_pipeline.py
```

is sufficient to regenerate the database from the source website.

---

## Design Summary

The design follows a simple ETL pattern:

```text
Books to Scrape
       │
       ▼
   Requests
       │
       ▼
 BeautifulSoup
       │
       ▼
 Raw DataFrame
       │
       ▼
 Cleaning / Validation
       │
       ▼
 GBP → INR
       │
       ▼
 Normalized SQLite
       │
       ├──────────────┐
       ▼              ▼
      SQL          pandas
       │              │
       └──────┬───────┘
              ▼
        Result Validation
```

The implementation deliberately uses a fixed conversion rate rather than an external currency API because the assignment defines 1 GBP = 105.50 INR as the required grading baseline.

## Currency Conversion

The project-defined fixed conversion rate is:

1 GBP = 105.50 INR

This is an artificial fixed baseline supplied by the assignment.
No live currency API is used.

price_inr = price_gbp × 105.50
