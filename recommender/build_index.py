import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os

# Use relative paths for deployment compatibility
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # go up from recommender/build_index.py to root
if os.path.basename(BASE_DIR) == 'recommender': # Safety check
     BASE_DIR = os.path.dirname(BASE_DIR)

# Fallback
if not os.path.exists(os.path.join(BASE_DIR, 'data')):
    BASE_DIR = os.getcwd()

INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "shl_catalogue_clean.csv")
INDEX_DIR = os.path.join(BASE_DIR, "data", "indexes")
INDEX_FILE = os.path.join(INDEX_DIR, "shl_embeddings.index")
META_FILE = os.path.join(INDEX_DIR, "shl_metadata.pkl")
MODEL_NAME = 'all-MiniLM-L6-v2'

def build_index():
    print("Loading data...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} records.")

    # Create content for embedding
    # Format: "Assessment: <Name>. Type: <Type>. Description: <Desc>"
    print("Preparing text chunks...")
    chunks = df.apply(
        lambda x: f"Assessment: {x['assessment_name']}. Type: {x['test_type']}. Description: {x['description']}", 
        axis=1
    ).tolist()

    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    # Normalize embeddings to use Inner Product for Cosine Similarity
    embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype('float32')

    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")

    print("Building FAISS index...")
    # IndexFlatIP implements Inner Product (Cosine freq when normalized)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"Index contains {index.ntotal} vectors.")

    print("Saving index and metadata...")
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    faiss.write_index(index, INDEX_FILE)
    
    # Save metadata (the dataframe) so we can retrieve details by ID
    with open(META_FILE, 'wb') as f:
        pickle.dump(df, f)

    print("Done.")

if __name__ == "__main__":
    build_index()
