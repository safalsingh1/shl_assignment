import re
import os
from collections import Counter
from recommender.query_processor import QueryProcessor
from recommender.search_service import SHLRetriever

class RecommendationEngine:
    def __init__(self):
        # Initialize components
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("WARNING: GEMINI_API_KEY not set. Query understanding will operate in fallback mode.")
        
        self.processor = QueryProcessor(api_key=api_key)
        self.retriever = SHLRetriever()

    def _calculate_skill_score(self, text, skills):
        """
        Calculate normalized overlap score between text and skills.
        """
        if not skills or not text:
            return 0.0
            
        text_lower = text.lower()
        matched_count = 0
        for skill in skills:
            # Simple word boundary matching
            # Escaping skill to prevent regex errors
            skill_pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(skill_pattern, text_lower):
                matched_count += 1
                
        # Score is fraction of requested skills found
        return matched_count / len(skills)

    def recommend(self, query, min_results=5, max_results=10):
        print(f"DEBUG: Processing query: '{query}'")
        
        # 1. Analyze Query
        analysis = self.processor.analyze(query)
        skills = analysis.get('skills', [])
        required_types = analysis.get('required_test_types', ['K', 'P'])
        
        print(f"DEBUG: Extracted Skills: {skills}")
        print(f"DEBUG: Required Types: {required_types}")

        # 2. Retrieve Candidates (get more than needed for re-ranking/balancing)
        # Using k=20 to have a pool
        candidates = self.retriever.search(query, top_k=20)
        
        # 3. Re-rank
        reranked_candidates = []
        for cand in candidates:
            # Vector score is already cosine similarity (or inner product normalized)
            vector_score = cand['score']
            
            # Skill score
            desc = cand.get('description', '')
            skill_score = self._calculate_skill_score(desc, skills)
            
            # Weighted Combination
            # Vector score for MiniLM/InnerProduct is roughly -1 to 1, usually 0.3-0.8 for matches.
            # Skill score is 0 to 1.
            final_score = (0.7 * vector_score) + (0.3 * skill_score)
            
            cand['final_score'] = final_score
            cand['skill_score'] = skill_score
            reranked_candidates.append(cand)
            
        # Sort by final score descending
        reranked_candidates.sort(key=lambda x: x['final_score'], reverse=True)

        # 4. Filter & Balance
        final_results = []
        
        # Split by type
        k_candidates = [c for c in reranked_candidates if c['test_type'] == 'K']
        p_candidates = [c for c in reranked_candidates if c['test_type'] == 'P']
        
        # Check requirements
        needs_k = 'K' in required_types
        needs_p = 'P' in required_types
        
        if needs_k and needs_p:
            # We need both.
            # Strategy: Take highest from K, highest from P, then fill rest with highest remaining
            if k_candidates:
                final_results.append(k_candidates.pop(0))
            if p_candidates:
                final_results.append(p_candidates.pop(0))
                
            # Pool rest and sort
            remaining = k_candidates + p_candidates
            remaining.sort(key=lambda x: x['final_score'], reverse=True)
            
            # Fill up to max_results (or at least min_results)
            needed = max_results - len(final_results)
            final_results.extend(remaining[:needed])
            
        elif needs_k:
            final_results = k_candidates[:max_results]
        elif needs_p:
            final_results = p_candidates[:max_results]
        else:
            # Should not happen based on logic, but default to top
            final_results = reranked_candidates[:max_results]
            
        # Ensure minimum if possible
        if len(final_results) < min_results:
            # Fallback: add whatever is left from candidates that match ANY requirement
            # Or just return what we have
            pass

        # 5. Format Output
        output = []
        for res in final_results:
            output.append({
                "assessment_name": res['assessment_name'],
                "assessment_url": res.get('assessment_url', ''), # Might be missing if old data
                "test_type": res['test_type']
            })
            
        return output

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    # Test
    engine = RecommendationEngine()
    q = "I need a sales manager who knows java"
    results = engine.recommend(q)
    
    import json
    print(f"\nFinal Recommendations ({len(results)}):")
    print(json.dumps(results, indent=2))
