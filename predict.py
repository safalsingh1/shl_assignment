import pandas as pd
import csv
import os
from recommender.recommendation_engine import RecommendationEngine
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = r"c:\Users\safal\Desktop\shl_assignment\data\test_queries.csv"
OUTPUT_FILE = r"c:\Users\safal\Desktop\shl_assignment\predictions.csv"

def generate_predictions():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print("Initializing engine...")
    try:
        engine = RecommendationEngine()
    except Exception as e:
        print(f"Error initializing engine: {e}")
        return

    print(f"Loading queries from {INPUT_FILE}...")
    try:
        queries_df = pd.read_csv(INPUT_FILE)
        if 'Query' not in queries_df.columns:
            print("Error: Input CSV must have a 'Query' column.")
            return
        queries = queries_df['Query'].tolist()
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    print(f"Processing {len(queries)} queries...")
    
    # Open output file for writing
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Query", "Assessment_url"])
        
        for q in queries:
            print(f"Predicting for: {q}")
            try:
                results = engine.recommend(q)
                for res in results:
                    url = res.get('assessment_url', '')
                    writer.writerow([q, url])
            except Exception as e:
                print(f"Error processing query '{q}': {e}")
                
    print(f"Predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_predictions()
