from rag.vectorstore import MathKnowledgeBase
from models.schemas import RetrievedContext, ParsedProblem

# ---------------------------
# RAG SAFETY THRESHOLD
# ---------------------------
MIN_RELEVANCE_SCORE = 0.35  # below this = unreliable context


class RAGRetriever:
    def __init__(self):
        self.kb = MathKnowledgeBase()
    
    def retrieve(
        self, 
        problem: ParsedProblem, 
        k: int = 5
    ) -> list[RetrievedContext]:
        """
        Retrieve relevant context for a problem.
        Returns an empty list if no reliable context is found.
        """

        # ---------------------------
        # Build query from problem
        # ---------------------------
        query_parts = [problem.problem_text]
        
        if problem.topic:
            query_parts.append(f"Topic: {problem.topic.value}")
        
        if problem.variables:
            query_parts.append(f"Variables: {', '.join(problem.variables)}")
        
        query = " ".join(query_parts)

        # ---------------------------
        # General retrieval
        # ---------------------------
        general_results = self.kb.search(query, k=k)

        # ---------------------------
        # Topic-specific retrieval
        # ---------------------------
        if problem.topic and problem.topic.value != "unknown":
            topic_results = self.kb.search_by_topic(
                problem.problem_text,
                problem.topic.value,
                k=2
            )

            # Merge + deduplicate
            seen_texts = {r.text for r in general_results}
            for r in topic_results:
                if r.text not in seen_texts:
                    general_results.append(r)

        # ---------------------------
        # Sort by relevance
        # ---------------------------
        general_results.sort(
            key=lambda x: x.relevance_score,
            reverse=True
        )

        # ---------------------------
        # RELEVANCE FILTER (CRITICAL)
        # ---------------------------
        filtered_results = [
            r for r in general_results
            if r.relevance_score >= MIN_RELEVANCE_SCORE
        ]

        # If nothing reliable → signal RAG failure
        if not filtered_results:
            return []

        return filtered_results[:k]
    
    def retrieve_for_verification(
        self, 
        problem: ParsedProblem, 
        solution: str
    ) -> list[RetrievedContext]:
        """
        Retrieve verification context (mistakes + formulas),
        filtered by relevance.
        """

        if not problem.topic:
            return []

        # ---------------------------
        # Retrieve verification material
        # ---------------------------
        mistakes = self.kb.get_common_mistakes(problem.topic.value)
        formulas = self.kb.get_formulas(problem.topic.value)

        combined = mistakes + formulas

        # ---------------------------
        # Filter weak signals
        # ---------------------------
        verified_context = [
            c for c in combined
            if c.relevance_score >= MIN_RELEVANCE_SCORE
        ]

        return verified_context
