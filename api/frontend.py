import streamlit as st
import requests
import pandas as pd

# Page Config
st.set_page_config(page_title="SHL Recommender", layout="wide")

import os
# Get API URL from env or streamlit secrets, default to localhost
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("SHL Assessment Recommender")
st.markdown(f"Enter a job description or query to get relevant SHL assessment recommendations.\\n*Connected to: `{API_URL}`*")
st.info("Note: The backend is hosted on Railway Free Tier and may sleep after inactivity. The first request might take 1-2 minutes to wake it up. If it fails, please wait a moment and try again. Subsequent requests will be fast.")


# Input
query = st.text_area("Job Description / Query", height=100, placeholder="e.g. Seeking a Senior Java Developer with leadership skills...")

# Submit
if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Analyzing and retrieving..."):
            try:
                # Call API
                response = requests.post(f"{API_URL}/recommend", json={"query": query})
                
                if response.status_code == 200:
                    data = response.json()
                    recs = data.get("recommended_assessments", [])
                    
                    if not recs:
                        st.info("No recommendations found.")
                    else:
                        st.success(f"Found {len(recs)} recommendations:")
                        
                        # Create DataFrame for display
                        df = pd.DataFrame(recs)
                        
                        # Rename columns for display
                        df = df.rename(columns={
                            "name": "Assessment Name",
                            "test_type": "Type",
                            "url": "URL",
                            "duration": "Duration (mins)",
                            "adaptive_support": "Adaptive",
                            "remote_support": "Remote"
                        })
                        
                        # Reorder
                        cols = ["Assessment Name", "Type", "Duration (mins)", "Adaptive", "Remote", "URL"]
                        # Ensure cols exist
                        final_cols = [c for c in cols if c in df.columns]
                        df = df[final_cols]
                        
                        # Display
                        st.table(df)
                        
                else:
                    st.error(f"Error from API: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API. Is 'uvicorn api:app' running?")
            except Exception as e:
                st.error(f"An error occurred: {e}")
