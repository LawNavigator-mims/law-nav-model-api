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
from supabase import create_client, Client



load_dotenv()


url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

# File paths
FAISS_INDEX_PATH = "vector_chunks_MiniLM.faiss"
DOCUMENT_METADATA_PATH = "vector_chunks_MiniLM_metadata.csv"

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
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load Hugging Face Token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# # Define the DeepSeek model
# llm_model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"

# # Load Tokenizer and Model with authentication
# llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, token=HF_TOKEN)
# llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto", token=HF_TOKEN)

# OpenAI API Client for Llama API
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")  # Stored the API key in render.com environment variables
client = OpenAI(
    api_key=LLAMA_API_KEY,
    base_url="https://api.llama-api.com"
)

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


def convert_np_values_to_native(d):
    """
    Convert NumPy dtypes in a dict to native Python types
    so they are JSON-serializable (e.g., int, float, list).
    """
    new_dict = {}
    for k, v in d.items():
        # If it's a NumPy integer (e.g., np.int64), convert to Python int
        if isinstance(v, np.integer):
            new_dict[k] = int(v)
        # If it's a NumPy float (e.g., np.float64), convert to Python float
        elif isinstance(v, np.floating):
            new_dict[k] = float(v)
        # If it's a NumPy array, convert to list
        elif isinstance(v, np.ndarray):
            new_dict[k] = v.tolist()
        else:
            new_dict[k] = v
    return new_dict

def extract_clean_answer(response_text):
    parts = response_text.split("Answer:", 1)
    if len(parts) < 2:
        return response_text
    answer = parts[1].strip()
    return re.sub(r"(Question:.*|Context:.*)", "", answer, flags=re.DOTALL).strip()

def search_documents(query, top_k=10):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    # if query_embedding.shape[1] != index.d:
    #     raise ValueError(f"Embedding dimension {query_embedding.shape[1]} does not match FAISS index dimension {index.d}")
    
    # distances, indices = index.search(query_embedding, top_k)

    
    args =  {
        'query_embedding': query_embedding[0],
        'match_threshold': 0.5,
        'match_count': 100
    }
    args = convert_np_values_to_native(args)
    response = supabase.rpc('match_documents', args).execute()

    results = []
    for row in response.data:
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
    return {"message": "Welcome to the Law Navigator API. Use /search/ to generate legal responses."}

@app.post("/search/")
def search_endpoint(request: QueryRequest, top_k: int = 5):
    response_text = generate_response(request.text)
    return {"query": request.text, "response": response_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
