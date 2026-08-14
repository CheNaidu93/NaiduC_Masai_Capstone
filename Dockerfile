FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docs ./docs
COPY models.py prompt.py ingest.py graph.py main.py ./

# Build the local ChromaDB index during image build.
RUN python ingest.py

ENV MOCK_LLM=1

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
