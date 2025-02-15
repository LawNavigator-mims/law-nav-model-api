from fastapi import FastAPI
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import uvicorn

# Load FAISS index and metadata
FAISS_INDEX_PATH = "vector_index.faiss"
DOCUMENT_METADATA_PATH = "document_metadata.csv"

index = faiss.read_index(FAISS_INDEX_PATH)
df_metadata = pd.read_csv(DOCUMENT_METADATA_PATH)

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

app = FastAPI()

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
        row = df_metadata.iloc[idx]
        results.append({
            "title": row["title"],
            "chapter": row["chapter"],
            "section": row["section"],
            "content": row["content"],
            "filename": row["filename"],
            "page": row["page"]
        })

    return {"query": request.text, "retrieved_docs": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)

