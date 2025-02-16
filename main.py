from fastapi import FastAPI
import faiss
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# File paths
FAISS_INDEX_PATH = "vector_index.faiss"
DOCUMENT_METADATA_PATH = "document_metadata.csv"

# Check if FAISS index exists
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"FAISS index file not found at {FAISS_INDEX_PATH}")

# Load FAISS index and metadata
index = faiss.read_index(FAISS_INDEX_PATH)
df_metadata = pd.read_csv(DOCUMENT_METADATA_PATH)

# Ensure metadata file has expected columns
required_columns = {"title", "chapter", "section", "content", "filename", "page"}
if not required_columns.issubset(df_metadata.columns):
    raise ValueError(f"Missing columns in metadata file. Expected: {required_columns}")

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class QueryRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Welcome to the Law Navigator API. Use /search/ or /generate-response/ endpoints."}

@app.post("/search/")
def search_documents(request: QueryRequest, top_k: int = 5):
    """Retrieves the most relevant documents based on the input query."""
    query_embedding = embedding_model.encode([request.text], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(df_metadata):  # Ensure index is within bounds
            continue

        row = df_metadata.iloc[idx]
        results.append({
            "title": str(row.get("title", "")),
            "chapter": str(row.get("chapter", "")),
            "section": str(row.get("section", "")),
            "content": str(row.get("content", "")),
            "filename": str(row.get("filename", "")),
            "page": int(row["page"]) if pd.notna(row["page"]) and isinstance(row["page"], (np.integer, np.int64)) else None
        })

    return {"query": request.text, "retrieved_docs": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)