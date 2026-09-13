from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DB = ROOT / "chroma_db"

COLLECTION_NAME = "zepto_policy_corpus"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents():
    records = []
    for path in sorted(DOCS.glob("doc_*.txt")):
        records.append(
            {
                "id": path.stem,
                "text": path.read_text(encoding="utf-8").strip(),
            }
        )
    if len(records) != 8:
        raise RuntimeError(f"Expected 8 documents, found {len(records)}")
    return records


def build_index():
    records = load_documents()

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB))

    # Rebuild deterministically so the script is safe to rerun.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"document_id": r["id"]} for r in records],
    )

    print(f"Indexed {collection.count()} documents.")
    print(f"ChromaDB path: {DB}")


if __name__ == "__main__":
    build_index()
