"""
Vector-based memory for semantic similarity search of past problems and solutions.
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import Optional
from datetime import datetime
import hashlib

from rag.embeddings import EmbeddingService
from config import Config


class VectorMemory:
    """
    Vector memory store for semantic search of past interactions.
    Enables finding similar problems and reusing solution patterns.
    """
    
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or str(Path(Config.CHROMA_PERSIST_DIR) / "memory")
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_service = EmbeddingService()
        
        # Collection for problems
        self.problems_collection = self.client.get_or_create_collection(
            name="problems",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Collection for solutions
        self.solutions_collection = self.client.get_or_create_collection(
            name="solutions",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Collection for error patterns (to avoid repeating mistakes)
        self.errors_collection = self.client.get_or_create_collection(
            name="error_patterns",
            metadata={"hnsw:space": "cosine"}
        )
    
    def _generate_id(self, text: str) -> str:
        """Generate a deterministic ID from text"""
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def store_problem(
        self,
        problem_text: str,
        topic: str,
        solution: str,
        interaction_id: int,
        feedback: Optional[str] = None
    ) -> str:
        """Store a problem and its solution in vector memory"""
        problem_id = f"prob_{interaction_id}"
        
        # Embed and store problem
        problem_embedding = self.embedding_service.embed_single(problem_text)
        
        self.problems_collection.upsert(
            ids=[problem_id],
            embeddings=[problem_embedding],
            documents=[problem_text],
            metadatas=[{
                "interaction_id": interaction_id,
                "topic": topic,
                "feedback": feedback or "",
                "timestamp": datetime.now().isoformat()
            }]
        )
        
        # Embed and store solution
        solution_id = f"sol_{interaction_id}"
        solution_embedding = self.embedding_service.embed_single(solution)
        
        self.solutions_collection.upsert(
            ids=[solution_id],
            embeddings=[solution_embedding],
            documents=[solution],
            metadatas=[{
                "interaction_id": interaction_id,
                "problem_id": problem_id,
                "topic": topic,
                "feedback": feedback or "",
                "timestamp": datetime.now().isoformat()
            }]
        )
        
        return problem_id
    
    def store_error_pattern(
        self,
        problem_text: str,
        wrong_solution: str,
        error_description: str,
        correct_solution: str,
        topic: str
    ) -> str:
        """Store an error pattern to help avoid similar mistakes"""
        error_id = self._generate_id(problem_text + wrong_solution)
        
        # Combine problem and wrong solution for embedding
        combined_text = f"Problem: {problem_text}\nWrong approach: {wrong_solution}"
        embedding = self.embedding_service.embed_single(combined_text)
        
        self.errors_collection.upsert(
            ids=[error_id],
            embeddings=[embedding],
            documents=[combined_text],
            metadatas=[{
                "topic": topic,
                "error_description": error_description,
                "correct_solution": correct_solution[:1000],  # Limit size
                "timestamp": datetime.now().isoformat()
            }]
        )
        
        return error_id
    
    def find_similar(
        self,
        query_text: str,
        k: int = 3,
        filter_topic: Optional[str] = None,
        filter_feedback: Optional[str] = None
    ) -> list[dict]:
        """Find similar problems from memory"""
        query_embedding = self.embedding_service.embed_single(query_text)
        
        # Build filter
        where_filter = {}
        if filter_topic:
            where_filter["topic"] = filter_topic
        if filter_feedback:
            where_filter["feedback"] = filter_feedback
        
        results = self.problems_collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter if where_filter else None
        )
        
        similar_problems = []
        
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                problem = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                
                # Get corresponding solution
                solution = self._get_solution_for_problem(metadata.get("interaction_id"))
                
                similar_problems.append({
                    "problem": problem,
                    "solution": solution,
                    "topic": metadata.get("topic"),
                    "feedback": metadata.get("feedback"),
                    "similarity": 1 - distance,  # Convert distance to similarity
                    "interaction_id": metadata.get("interaction_id")
                })
        
        return similar_problems
    
    def find_similar_solutions(
        self,
        solution_approach: str,
        k: int = 3,
        filter_topic: Optional[str] = None
    ) -> list[dict]:
        """Find similar solution approaches"""
        query_embedding = self.embedding_service.embed_single(solution_approach)
        
        where_filter = {"topic": filter_topic} if filter_topic else None
        
        results = self.solutions_collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter
        )
        
        similar_solutions = []
        
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                similar_solutions.append({
                    "solution": results["documents"][0][i],
                    "topic": results["metadatas"][0][i].get("topic"),
                    "feedback": results["metadatas"][0][i].get("feedback"),
                    "similarity": 1 - results["distances"][0][i]
                })
        
        return similar_solutions
    
    def find_relevant_errors(
        self,
        problem_text: str,
        k: int = 3
    ) -> list[dict]:
        """Find relevant error patterns to avoid"""
        query_embedding = self.embedding_service.embed_single(problem_text)
        
        results = self.errors_collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        errors = []
        
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                errors.append({
                    "error_context": results["documents"][0][i],
                    "error_description": results["metadatas"][0][i].get("error_description"),
                    "correct_solution": results["metadatas"][0][i].get("correct_solution"),
                    "relevance": 1 - results["distances"][0][i]
                })
        
        return errors
    
    def _get_solution_for_problem(self, interaction_id: int) -> Optional[str]:
        """Get the solution associated with a problem"""
        if not interaction_id:
            return None
        
        solution_id = f"sol_{interaction_id}"
        
        try:
            result = self.solutions_collection.get(
                ids=[solution_id],
                include=["documents"]
            )
            
            if result["documents"]:
                return result["documents"][0]
        except Exception:
            pass
        
        return None
    
    def update_feedback(self, interaction_id: int, feedback: str):
        """Update the feedback for a stored problem/solution"""
        problem_id = f"prob_{interaction_id}"
        solution_id = f"sol_{interaction_id}"
        
        # Update problem metadata
        try:
            existing = self.problems_collection.get(ids=[problem_id])
            if existing["metadatas"]:
                metadata = existing["metadatas"][0]
                metadata["feedback"] = feedback
                self.problems_collection.update(
                    ids=[problem_id],
                    metadatas=[metadata]
                )
        except Exception:
            pass
        
        # Update solution metadata
        try:
            existing = self.solutions_collection.get(ids=[solution_id])
            if existing["metadatas"]:
                metadata = existing["metadatas"][0]
                metadata["feedback"] = feedback
                self.solutions_collection.update(
                    ids=[solution_id],
                    metadatas=[metadata]
                )
        except Exception:
            pass
    
    def get_topic_statistics(self) -> dict:
        """Get statistics about stored problems by topic"""
        stats = {}
        
        # Get all problems
        all_problems = self.problems_collection.get(include=["metadatas"])
        
        if all_problems["metadatas"]:
            for metadata in all_problems["metadatas"]:
                topic = metadata.get("topic", "unknown")
                feedback = metadata.get("feedback", "none")
                
                if topic not in stats:
                    stats[topic] = {"total": 0, "correct": 0, "incorrect": 0, "partial": 0, "none": 0}
                
                stats[topic]["total"] += 1
                stats[topic][feedback if feedback in ["correct", "incorrect", "partial"] else "none"] += 1
        
        return stats
    
    def clear_old_entries(self, days: int = 30):
        """Clear entries older than specified days"""
        from datetime import datetime, timedelta
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get old problem IDs
        all_problems = self.problems_collection.get(include=["metadatas"])
        
        old_ids = []
        if all_problems["metadatas"]:
            for i, metadata in enumerate(all_problems["metadatas"]):
                timestamp = metadata.get("timestamp", "")
                if timestamp and timestamp < cutoff_date:
                    old_ids.append(all_problems["ids"][i])
        
        if old_ids:
            self.problems_collection.delete(ids=old_ids)
            
            # Also delete corresponding solutions
            solution_ids = [id.replace("prob_", "sol_") for id in old_ids]
            self.solutions_collection.delete(ids=solution_ids)
        
        return len(old_ids)
    
    def export_to_training_data(self) -> list[dict]:
        """Export successful interactions for potential fine-tuning"""
        training_data = []
        
        all_problems = self.problems_collection.get(include=["documents", "metadatas"])
        
        if all_problems["documents"]:
            for i, problem in enumerate(all_problems["documents"]):
                metadata = all_problems["metadatas"][i]
                
                if metadata.get("feedback") == "correct":
                    solution = self._get_solution_for_problem(metadata.get("interaction_id"))
                    
                    if solution:
                        training_data.append({
                            "problem": problem,
                            "solution": solution,
                            "topic": metadata.get("topic")
                        })
        
        return training_data