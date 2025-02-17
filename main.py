from fastapi import FastAPI
import faiss
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# File paths
FAISS_INDEX_PATH = "vector_index.faiss"
DOCUMENT_METADATA_PATH = "document_metadata.csv"

# Ensure FAISS index exists
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

# Load Hugging Face Token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# Define the DeepSeek model
llm_model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"

# Load Tokenizer and Model with authentication
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, token=HF_TOKEN)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto", token=HF_TOKEN)

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
    return {"message": "Welcome to the Law Navigator API. Use /search/ to generate legal responses."}

@app.post("/search/")
def generate_response(request: QueryRequest, top_k: int = 5):
    """Retrieves the most relevant documents and generates a response."""
    query_embedding = embedding_model.encode([request.text], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    # Extract relevant chunks
    context_chunks = []
    for idx in indices[0]:
        if 0 <= idx < len(df_metadata):  # Ensure index is valid
            row = df_metadata.iloc[idx]
            context_chunks.append(str(row.get("content", "")))

    context = "\n".join(context_chunks) if context_chunks else "No relevant legal information found."

    # Generate response using DeepSeek LLM
    prompt = f"""You are an expert in legal regulations. Answer the user's question based on the given context.
If the context does not provide a complete answer, say "Not enough information in the provided context."
Use structured bullet points if multiple regulations apply.

### Context:
{context}

### Question:
{request.text}

### Response:
"""

    inputs = llm_tokenizer(prompt, return_tensors="pt").to("cuda")
    output = llm_model.generate(**inputs, max_length=512)
    response_text = llm_tokenizer.decode(output[0], skip_special_tokens=True)

    return {"query": request.text, "response": response_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
