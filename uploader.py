import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv  
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from supabase.client import ClientOptions

load_dotenv()


url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key,options=ClientOptions(
        postgrest_client_timeout=999999,
        storage_client_timeout=10,
        schema="public",
    ))

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN = os.environ.get("HF_TOKEN")


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


def upload_documents():
    df = pd.read_csv('vector_chunks_MiniLM_metadata.csv')
    rows = df.iloc[2:]
    data=[]

    for index,entries in rows.iterrows():
    
        query_embedding = embedding_model.encode([entries['content']], convert_to_numpy=True)
        document_data = {
            "title": entries['title'],
            "chapter": entries['chapter'],
            "section": entries['section'],
            "content": entries['content'],
            "filename": entries['filename'],
            "page": entries['page'],
        
        }

        # Filter out NaN while allowing arrays
        filtered_data = {
            key: value for key, value in document_data.items()
            if (not isinstance(value, float) or not np.isnan(value))
            or isinstance(value, list)
        }

        # Convert embedding array into a list
        filtered_data['embedding'] = query_embedding[0]  # shape (384,) presumably

        # Convert all NumPy types to native Python types
        filtered_data = convert_np_values_to_native(filtered_data)
        data.append(filtered_data)

    # Insert into Supabase
    print(f"about to upload {len(data)} entries")
    BATCH_SIZE = 100

    for start in range(0, len(data), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_data = data[start:end]
        
        response = supabase.table("documents").insert(batch_data).execute()
        print(f"Inserted rows {start} to {end}, response: {response}")

def create_embeddings_vector():
    question="What are the key regulations on property tax in Sierra_Madre?"
    query_embedding = embedding_model.encode([question], convert_to_numpy=True)
    args =  {
        'query_embedding': query_embedding[0],
        'match_threshold': 0.5,
        'match_count': 100
    }
    args = convert_np_values_to_native(args)
    response = supabase.rpc('match_documents', args).execute()
    print(response)

create_embeddings_vector()