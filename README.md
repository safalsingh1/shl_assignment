# SHL Assessment Recommendation System

## Overview
This project implements an intelligent recommendation system for SHL assessments. It solves the problem of efficiently mapping natural language job descriptions or queries to specific assessments in the SHL catalogue using a Retrieval-Augmented Generation (RAG) approach with semantic re-ranking.

## Directory Structure
The codebase is organized as follows:
```
shl_assignment/
├── api/                    # API and Frontend
│   ├── api.py              # FastAPI application (Endpoints: /health, /recommend)
│   ├── frontend.py         # Streamlit UI
│   └── __init__.py
├── recommender/            # Core Engine Logic
│   ├── recommendation_engine.py  # Main orchestration (Retrieval + Re-ranking + Balancing)
│   ├── query_processor.py        # Gemini-based intent extraction
│   ├── search_service.py         # FAISS vector search
│   ├── build_index.py            # Index generation script
│   └── __init__.py
├── scraper/                # Data Acquisition
│   ├── scrape_shl.py       # Selenium scraper
│   ├── clean_shl.py        # Data cleaning pipeline
│   └── __init__.py
├── data/                   # Data Storage
│   ├── raw/                # Scraped CSVs
│   ├── processed/          # Cleaned CSVs
│   └── labeled/            # Training/Validation data
├── evaluation/             # Metrics & Reports
│   └── results.json
├── predict.py              # CLI Prediction Script (Entry point for CSV generation)
├── evaluate.py             # CLI Evaluation Script
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
```

## Setup
1.  **Prerequisites**: Python 3.9+
2.  **Environment**:
    - Rename `,env` to `.env` (already done).
    - Ensure `GEMINI_API_KEY` is set in `.env`.
3.  **Installation**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Web Application (Frontend)
The easiest way to test the system is via the Streamlit UI.
**Terminal 1 (Start API)**:
```bash
python -m uvicorn api.api:app --reload
```
**Terminal 2 (Start Frontend)**:
```bash
python -m streamlit run api/frontend.py
```
*Note: The frontend connects to `http://localhost:8000`.*

### 2. Generate Predictions (CSV)
To generate the `predictions.csv` file for the test set:
```bash
python predict.py
```
*Output*: `predictions.csv` (Columns: `Query`, `Assessment_url`)

### 3. Run Evaluation
To measure Recall@10 on the labeled dataset:
```bash
python evaluate.py
```

## Technical Approach
1.  **Data Ingestion**: Scraped ~380 assessments from SHL. Cleaned and normalized text.
2.  **Retrieval**: `sentence-transformers/all-MiniLM-L6-v2` embeddings indexed in `FAISS` for fast semantic search.
3.  **Query Understanding**: Google Gemini LLM extracts:
    - **Skills**: Hard/Soft skills (e.g., "Python", "Leadership").
    - **Test Types**: 'K' (Knowledge) or 'P' (Personality).
4.  **Re-ranking & Balancing**:
    - **Hybrid Score**: `0.7 * Vector_Sim + 0.3 * Skill_Overlap`.
    - **Balancing**: Enforces a mix of Knowledge and Personality tests if the query implies both.
5.  **Compliance**: 
    - Output matches strict API schema (`url`, `name`, `adaptive_support`, `description`, `duration`, `remote_support`, `test_type`).
    - `test_type` is returned as a list of strings.

## Metrics
- **Mean Recall@10**: 0.75 (Baseline) -> 0.75 (Re-ranked).
- *Note*: Tested on synthetic data. Real-world performance gains from re-ranking are expected to be higher on complex queries.

## Future Work
- Move to a persistent Vector DB (Chroma/Qdrant).
- Fine-tune embedding model on HR-specific corpus.
- Implement strictly structured output JSON parsing for Gemini to improve robustness.
