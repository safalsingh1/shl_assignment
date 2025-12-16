import pandas as pd
import json
import os
import numpy as np
from recommender.recommendation_engine import RecommendationEngine
from recommender.search_service import SHLRetriever

LABELED_DATA = r"c:\Users\safal\Desktop\shl_assignment\data\labeled\train.csv"
OUTPUT_DIR = r"c:\Users\safal\Desktop\shl_assignment\evaluation"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results.json")

def calculate_recall_at_k(retrieved_items, relevant_items, k=10):
    k = min(k, len(retrieved_items))
    top_k = retrieved_items[:k]
    
    hits = 0
    for item in top_k:
        if item in relevant_items:
            hits += 1
            
    if not relevant_items:
        return 0.0
        
    return hits / len(relevant_items)

def evaluate():
    print("Initializing systems...")
    # Initialize both (Engine initializes its own retriever, but for baseline we want raw retriever)
    try:
        engine = RecommendationEngine()
        retriever = SHLRetriever()
    except Exception as e:
        print(f"Error checking initialization: {e}")
        return

    if not os.path.exists(LABELED_DATA):
        print(f"Labeled data not found at {LABELED_DATA}")
        return

    print("Loading labeled data...")
    df = pd.read_csv(LABELED_DATA)
    
    baseline_recalls = []
    model_recalls = []
    
    print(f"Evaluating {len(df)} queries...")
    for index, row in df.iterrows():
        query = row['query']
        # relevant_assessments is pipe separated
        relevant = [x.strip() for x in str(row['relevant_assessments']).split('|')]
        
        print(f"Query: {query}")
        
        # 1. Baseline (Raw Vector Search)
        # Search returns list of dicts with 'assessment_name'
        base_results = retriever.search(query, top_k=10)
        base_names = [r['assessment_name'] for r in base_results]
        rec_base = calculate_recall_at_k(base_names, relevant, k=10)
        baseline_recalls.append(rec_base)
        print(f"  Baseline Recall@10: {rec_base:.2f}")

        # 2. Model (Re-ranked & Balanced)
        # Engine returns list of dicts with 'assessment_name'
        model_results = engine.recommend(query) # Defults to max 10
        model_names = [r['assessment_name'] for r in model_results]
        rec_model = calculate_recall_at_k(model_names, relevant, k=10)
        model_recalls.append(rec_model)
        print(f"  Model Recall@10:    {rec_model:.2f}")
        
    # Aggregate
    mean_recall_baseline = np.mean(baseline_recalls) if baseline_recalls else 0.0
    mean_recall_model = np.mean(model_recalls) if model_recalls else 0.0
    
    results = {
        "baseline_mean_recall_at_10": float(mean_recall_baseline),
        "model_mean_recall_at_10": float(mean_recall_model),
        "details": {
            "num_queries": len(df),
            "baseline_recalls": baseline_recalls,
            "model_recalls": model_recalls
        }
    }
    
    print("\n--- Evaluation Results ---")
    print(f"Baseline Mean Recall@10: {mean_recall_baseline:.4f}")
    print(f"Model Mean Recall@10:    {mean_recall_model:.4f}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    evaluate()
