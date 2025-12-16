import os
import json
import google.generativeai as genai
from typing import Dict, List, Any

class QueryProcessor:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            # We don't raise here to allow instantiation, but methods will fail or warn
            print("WARNING: GEMINI_API_KEY not found in environment variables.")
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyzes the query to extract skills and required test types.
        Returns a dictionary with 'skills' and 'required_test_types'.
        """
        if not self.api_key:
            return {"error": "API key missing", "skills": [], "required_test_types": ["K", "P"]}

        prompt = f"""
        You are a data extraction assistant for an assessment catalogue.
        Analyze the following user query to determine:
        1. Key skills or topics mentioned (e.g., "Java", "Sales", "Communication").
        2. The type of tests required based on the intent:
           - 'K' for Knowledge/Skill (coding, technical, aptitude, hard skills).
           - 'P' for Personality/Behavior (soft skills, culture fit, leadership, traits).
           - If unsure or both are implied, include both 'K' and 'P'.

        Query: "{query}"

        Return ONLY a valid JSON object with the following schema:
        {{
          "skills": ["string", "string"],
          "required_test_types": ["K", "P"]
        }}
        Do not include markdown formatting like ```json ... ```. Just the raw JSON string.
        """

        try:
            response = self.model.generate_content(prompt)
            text = response.text.replace('```json', '').replace('```', '').strip()
            data = json.loads(text)
            
            # Enforce schema validity
            skills = data.get("skills", [])
            if isinstance(skills, str): skills = [skills]
            
            types = data.get("required_test_types", [])
            if isinstance(types, str): types = [types]
            
            # Normalize types
            valid_types = []
            for t in types:
                if t.upper() in ['K', 'P']:
                    valid_types.append(t.upper())
            
            # If no valid types found, default to both to be safe
            if not valid_types:
                valid_types = ['K', 'P']
                
            return {
                "skills": [str(s) for s in skills],
                "required_test_types": list(set(valid_types))
            }
            
        except Exception as e:
            print(f"Error analyzing query: {e}")
            # Fallback
            return {
                "skills": [query], 
                "required_test_types": ["K", "P"]
            }

if __name__ == "__main__":
    # Test
    processor = QueryProcessor()
    if not processor.api_key:
        print("Skipping test: No API Key")
    else:
        q = "I need a Java developer with good team leading skills"
        print(f"Query: {q}")
        print(json.dumps(processor.analyze(q), indent=2))
        
        q2 = "personality test for sales"
        print(f"\nQuery: {q2}")
        print(json.dumps(processor.analyze(q2), indent=2))
