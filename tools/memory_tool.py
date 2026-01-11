from agno.tools import tool
from memory.store import MemoryStore
from config import Config

# Global memory instance
_memory = None

def get_memory():
    global _memory
    if _memory is None:
        _memory = MemoryStore(Config.MEMORY_DB_PATH)
    return _memory

@tool
def search_similar_problems(problem_text: str, num_results: int = 3) -> str:
    """
    Search for similar problems that have been solved before.
    
    Args:
        problem_text: The math problem to find similar problems for
        num_results: Number of similar problems to return
    
    Returns:
        Similar problems and their solutions from memory
    """
    memory = get_memory()
    
    similar = memory.find_similar_problems(problem_text, k=num_results)
    
    if not similar:
        return "No similar problems found in memory."
    
    output = ["Similar problems from memory:"]
    for i, entry in enumerate(similar, 1):
        output.append(f"\n[{i}] Problem: {entry['problem'][:200]}...")
        output.append(f"    Solution: {entry['solution'][:200]}...")
        output.append(f"    Feedback: {entry.get('feedback', 'No feedback')}")
        output.append(f"    Similarity: {entry['similarity']:.2f}")
    
    return "\n".join(output)

@tool
def get_successful_solutions(topic: str, limit: int = 5) -> str:
    """
    Get successful solutions for a topic that received positive feedback.
    
    Args:
        topic: Math topic to get solutions for
        limit: Maximum number of solutions to return
    
    Returns:
        Successful solutions from memory
    """
    memory = get_memory()
    
    solutions = memory.get_successful_solutions(topic, limit=limit)
    
    if not solutions:
        return f"No successful solutions found for {topic}"
    
    output = [f"Successful solutions for {topic}:"]
    for i, sol in enumerate(solutions, 1):
        output.append(f"\n[{i}] {sol['problem'][:150]}...")
        output.append(f"    → {sol['solution'][:150]}...")
    
    return "\n".join(output)