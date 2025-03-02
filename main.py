from fastapi import FastAPI
import faiss
import os
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
from io import StringIO
from google.cloud import storage, secretmanager

# Set the path to the service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/law-nav-sa-key.json"


#Load environment variables from .env (for local testing)
if os.getenv("ENV", "local") == "local":
    load_dotenv()  # Only load .env if running locally
    print("Running in local mode. Loaded .env file.")
else:
    print("Running in GCP mode. Using GCP secrets.")

# GCP Configuration
GCP_PROJECT_ID = "capstone-mims"
BUCKET_NAME = "rag-pipeline-storage"

# Initialize GCP clients
storage_client = storage.Client()
secret_client = secretmanager.SecretManagerServiceClient()

def get_secret(secret_name):
    """Retrieve secrets from Google Cloud Secret Manager."""
    secret_path = f"projects/{GCP_PROJECT_ID}/secrets/{secret_name}/versions/latest"
    response = secret_client.access_secret_version(name=secret_path)
    return response.payload.data.decode("UTF-8")

# Load secrets from GCP
HF_TOKEN = get_secret("hf-token")
LLAMA_API_KEY = get_secret("llama-api-key")

# #When testing locally
# # Download files from GCS
# def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
#     """Download a file from GCS to local storage."""
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#     blob.download_to_filename(destination_file_name)
#     print(f"Downloaded {source_blob_name} to {destination_file_name}")

# # File paths
# FAISS_INDEX_PATH = "vector_index_2_temp.faiss"
# DOCUMENT_METADATA_PATH = "document_metadata_2_temp.csv"

# # Download index and metadata from GCS
# download_from_gcs(BUCKET_NAME, "embeddings/vector_chunks_MiniLM.faiss", FAISS_INDEX_PATH)
# download_from_gcs(BUCKET_NAME, "embeddings/vector_chunks_MiniLM_metadata.csv", DOCUMENT_METADATA_PATH)

# # Load FAISS index and metadata
# index = faiss.read_index(FAISS_INDEX_PATH)
# df_metadata = pd.read_csv(DOCUMENT_METADATA_PATH)
# # end testing locally 

# Access files directly from GCS (no local download)
def load_gcs_file_as_bytes(bucket_name, blob_name):
    """Read a file from GCS as bytes (for FAISS and CSV loading)."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()

# Load FAISS index and metadata directly from GCS
faiss_index_bytes = load_gcs_file_as_bytes(BUCKET_NAME, "embeddings/vector_index_2.faiss")
with open("/dev/shm/vector_index_2.faiss", "wb") as f:
    f.write(faiss_index_bytes)
index = faiss.read_index("/dev/shm/vector_index_2.faiss")

metadata_csv_bytes = load_gcs_file_as_bytes(BUCKET_NAME, "embeddings/document_metadata_2.csv")
df_metadata = pd.read_csv(StringIO(metadata_csv_bytes.decode("utf-8")))

# Ensure metadata file has expected columns
required_columns = {"title", "chapter", "section", "content", "filename", "page"}
if not required_columns.issubset(df_metadata.columns):
    raise ValueError(f"Missing columns in metadata file. Expected: {required_columns}")

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize Llama API client
client = OpenAI(api_key=LLAMA_API_KEY, base_url="https://api.llama-api.com")

# DeepSeek Model Setup (Commented Out for future use)
# llm_model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
# llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, token=HF_TOKEN)
# llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto", token=HF_TOKEN)

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

def extract_clean_answer(response_text):
    """Clean the LLM response to extract only the answer part."""
    parts = response_text.split("Answer:", 1)
    if len(parts) < 2:
        return response_text
    answer = parts[1].strip()
    return re.sub(r"(Question:.*|Context:.*)", "", answer, flags=re.DOTALL).strip()

def search_documents(query, top_k=10):
    """Search relevant documents using FAISS index."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(df_metadata):
            row = df_metadata.iloc[idx]
            results.append({
                "title": row["title"],
                "chapter": row["chapter"],
                "section": row["section"],
                "content": row["content"],
                "filename": row["filename"],
                "page": row["page"]
            })

    return pd.DataFrame(results)

def generate_response(query):
    """Generate LLM response using retrieved documents."""
    retrieved_docs = search_documents(query)
    context = "\n".join(retrieved_docs["content"].tolist())[:1500]

    prompt = f"""
You are an expert in legal regulations. Your task is to summarize the key property tax regulations
from the provided legal documents and present them in a clear and structured format.

## **Summary of Key Regulations on Property Tax**
- First, provide a **brief summary** (2-3 sentences) of the key points.
- Then, list the main regulations as **bullet points**, clearly stating the jurisdiction, chapter, and section.
- Finally, include the **source citations** in brackets `[Source: <Jurisdiction>, Title <X>, Chapter <Y>]`.

Context:
{context}

Question: {query}
Answer:
"""

    # Llama API call
    response = client.chat.completions.create(
        model="llama3.1-70b",
        messages=[
            {"role": "system", "content": "Assistant is a large language model trained to provide legal insights."},
            {"role": "user", "content": prompt}
        ]
    )

    return extract_clean_answer(response.choices[0].message.content)

@app.get("/")
def home():
    """Validity check endpoint."""
    return {"message": "Welcome to the Law Navigator API. Use /search/ to generate legal responses."}

@app.post("/search/")
def search_endpoint(request: QueryRequest, top_k: int = 5):
    """Endpoint to handle search queries."""
    response_text = generate_response(request.text)
    return {"query": request.text, "response": response_text}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Default to 8080 for Cloud Run
    uvicorn.run(app, host="0.0.0.0", port=port)