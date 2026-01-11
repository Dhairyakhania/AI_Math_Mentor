import chromadb
from chromadb.config import Settings
from config import Config
from rag.embeddings import EmbeddingService
from models.schemas import RetrievedContext
from typing import Optional
import hashlib

class VectorStore:
    def __init__(self, collection_name: str = "math_knowledge"):
        self.client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_service = EmbeddingService()
    
    def add_documents(self, documents: list[dict]) -> list[str]:
        """
        Add documents to vector store.
        documents: [{"text": str, "metadata": dict}]
        """
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            # Generate deterministic ID
            doc_id = hashlib.md5(doc["text"].encode()).hexdigest()[:12]
            ids.append(doc_id)
            texts.append(doc["text"])
            metadatas.append(doc.get("metadata", {}))
        
        # Generate embeddings
        embeddings = self.embedding_service.embed(texts)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        return ids
    
    def search(
        self, 
        query: str, 
        k: int = 3, 
        filter_metadata: Optional[dict] = None
    ) -> list[RetrievedContext]:
        """Search for relevant documents"""
        query_embedding = self.embedding_service.embed_single(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_metadata
        )
        
        contexts = []
        for i in range(len(results["documents"][0])):
            contexts.append(RetrievedContext(
                text=results["documents"][0][i],
                source=results["metadatas"][0][i].get("source", "unknown"),
                relevance_score=1 - results["distances"][0][i],  # Convert distance to similarity
                metadata=results["metadatas"][0][i]
            ))
        
        return contexts
    
    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(self.collection.name)


class MathKnowledgeBase(VectorStore):
    """Specialized vector store for math knowledge"""
    
    def __init__(self):
        super().__init__(collection_name="math_knowledge")
    
    def search_by_topic(self, query: str, topic: str, k: int = 3) -> list[RetrievedContext]:
        """Search within a specific topic"""
        return self.search(query, k=k, filter_metadata={"topic": topic})
    
    def get_formulas(self, topic: str) -> list[RetrievedContext]:
        """Get formulas for a topic"""
        return self.search(f"{topic} formulas", k=5, filter_metadata={"type": "formula"})
    
    def get_common_mistakes(self, topic: str) -> list[RetrievedContext]:
        """Get common mistakes for a topic"""
        return self.search(
            f"{topic} common mistakes pitfalls", 
            k=3, 
            filter_metadata={"type": "common_mistakes"}
        )