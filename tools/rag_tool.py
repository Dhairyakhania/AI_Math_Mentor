from agno.tools import tool
from rag.retriever import RAGRetriever
from models.schemas import ParsedProblem, MathTopic

_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


@tool
def search_knowledge_base(query: str, topic: str = None, num_results: int = 3) -> str:
    retriever = get_retriever()

    problem = ParsedProblem(
        problem_text=query,
        topic=MathTopic(topic) if topic else MathTopic.UNKNOWN
    )

    results = retriever.retrieve(problem, k=num_results)

    if not results:
        return "[]"

    return str([
        {
            "source": r.source,
            "text": r.text,
            "score": r.relevance_score
        }
        for r in results
    ])


@tool
def get_formulas_for_topic(topic: str) -> str:
    retriever = get_retriever()
    results = retriever.kb.get_formulas(topic)
    return str([r.text for r in results]) if results else "[]"


@tool
def get_common_mistakes(topic: str) -> str:
    retriever = get_retriever()
    results = retriever.kb.get_common_mistakes(topic)
    return str([r.text for r in results]) if results else "[]"
