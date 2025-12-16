from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic import BaseModel
from typing import List
import sys
import os

# Add root to path so we can import recommender
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommender.recommendation_engine import RecommendationEngine
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Initialize Engine
# Ensure GEMINI_API_KEY is in env or this will use fallback
try:
    engine = RecommendationEngine()
except Exception as e:
    print(f"Error initializing engine: {e}")
    engine = None

class RecommendationInput(BaseModel):
    query: str

class RecommendationItem(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationOutput(BaseModel):
    recommended_assessments: List[RecommendationItem]

@app.get("/health")
def health_check():
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendationOutput)
def recommend(input_data: RecommendationInput):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        results = engine.recommend(input_data.query)
        
        filtered_results = []
        for res in results:
            # Map Test Type
            tt = res.get('test_type', 'K')
            type_list = ["Knowledge & Skills"] if tt == 'K' else ["Personality & Behavior"]
            
            # Extract Duration (dummy logic or regex)
            import re
            desc = res.get('description', '')
            dur_match = re.search(r'(\d+)\s*(?:min|minute)', desc, re.IGNORECASE)
            duration = int(dur_match.group(1)) if dur_match else 30 # Default to 30 if not found
            
            filtered_results.append(RecommendationItem(
                url=res.get('assessment_url', ''),
                name=res['assessment_name'],
                adaptive_support="Yes", # Defaulting as data not available
                description=desc,
                duration=duration,
                remote_support="Yes", # Defaulting
                test_type=type_list
            ))
            
        return RecommendationOutput(recommended_assessments=filtered_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
