import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

INDEX_DIR = r"c:\Users\safal\Desktop\shl_assignment\data\indexes"
INDEX_FILE = os.path.join(INDEX_DIR, "shl_embeddings.index")
META_FILE = os.path.join(INDEX_DIR, "shl_metadata.pkl")
MODEL_NAME = 'all-MiniLM-L6-v2'

class SHLRetriever:
    def __init__(self, index_path=INDEX_FILE, meta_path=META_FILE, model_name=MODEL_NAME):
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Index or Metadata file not found. Run build_index.py first.")
            
        print(f"Loading index from {index_path}...")
        self.index = faiss.read_index(index_path)
        
        print(f"Loading metadata from {meta_path}...")
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)
            
        print(f"Loading model {model_name}...")
        self.model = SentenceTransformer(model_name)
        
    def search(self, query, top_k=5):
        """
        Search for assessments matching the query.
        Returns a list of dictionaries with assessment details and score.
        """
        # Encode query
        query_vector = self.model.encode([query], normalize_embeddings=True)
        query_vector = np.array(query_vector).astype('float32')
        
        # Search index
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue # Should not happen in Flat index unless k > n
            
            row = self.metadata.iloc[idx]
            results.append({
                'score': float(score),
                'assessment_name': row['assessment_name'],
                'test_type': row['test_type'],
                'description': row['description'],
                'assessment_url': row['assessment_url'] if 'assessment_url' in row else '',
                'category_tags': row['category_tags'] if 'category_tags' in row else ''
            })
            
        return results

if __name__ == "__main__":
    # Test run
    try:
        retriever = SHLRetriever()
        test_query = "sales manager"
        print(f"\nSearching for '{test_query}'...")
        results = retriever.search(test_query)
        for r in results:
            print(f"[{r['score']:.4f}] {r['assessment_name']} ({r['test_type']})")
    except Exception as e:
        print(f"Error: {e}")
