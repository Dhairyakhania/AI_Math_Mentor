from rag.vectorstore import MathKnowledgeBase
from models.schemas import RetrievedContext, ParsedProblem
from typing import Optional

class RAGRetriever:
    def __init__(self):
        self.kb = MathKnowledgeBase()
    
    def retrieve(
        self, 
        problem: ParsedProblem, 
        k: int = 5
    ) -> list[RetrievedContext]:
        """Retrieve relevant context for a problem"""
        # Build query from problem
        query_parts = [problem.problem_text]
        
        if problem.topic:
            query_parts.append(f"Topic: {problem.topic.value}")
        
        if problem.variables:
            query_parts.append(f"Variables: {', '.join(problem.variables)}")
        
        query = " ".join(query_parts)
        
        # Retrieve from general search
        general_results = self.kb.search(query, k=k)
        
        # Retrieve topic-specific if topic is known
        if problem.topic and problem.topic.value != "unknown":
            topic_results = self.kb.search_by_topic(
                problem.problem_text, 
                problem.topic.value, 
                k=2
            )
            # Merge and deduplicate
            seen_texts = {r.text for r in general_results}
            for r in topic_results:
                if r.text not in seen_texts:
                    general_results.append(r)
        
        # Sort by relevance
        general_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return general_results[:k]
    
    def retrieve_for_verification(
        self, 
        problem: ParsedProblem, 
        solution: str
    ) -> list[RetrievedContext]:
        """Retrieve context for verification"""
        # Get common mistakes
        mistakes = self.kb.get_common_mistakes(problem.topic.value)
        
        # Get relevant formulas to check
        formulas = self.kb.get_formulas(problem.topic.value)
        
        return mistakes + formulas