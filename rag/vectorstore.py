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

        # Using cosine distance (0 = identical, 1 = orthogonal)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.embedding_service = EmbeddingService()
    
    def add_documents(self, documents: list[dict]) -> list[str]:
        """
        Add documents to vector store.

        documents: [
            {
                "text": str,
                "metadata": {
                    "source": str,
                    "topic": str,
                    "type": str
                }
            }
        ]
        """

        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            if "text" not in doc or not doc["text"].strip():
                continue

            # Deterministic ID to avoid duplicates
            doc_id = hashlib.md5(doc["text"].encode()).hexdigest()[:12]

            metadata = doc.get("metadata", {})

            # ---- METADATA ENFORCEMENT (CRITICAL) ----
            enforced_metadata = {
                "source": metadata.get("source", "manual_kb"),
                "topic": metadata.get("topic", "unknown"),
                "type": metadata.get("type", "general")
            }

            ids.append(doc_id)
            texts.append(doc["text"])
            metadatas.append(enforced_metadata)
        
        if not texts:
            return []

        # Generate embeddings
        embeddings = self.embedding_service.embed(texts)
        
        # Add to Chroma collection
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
        """
        Search for relevant documents.

        Chroma returns cosine distance ∈ [0, 1]
        Similarity is computed as (1 - distance).
        """

        query_embedding = self.embedding_service.embed_single(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_metadata
        )
        
        contexts = []

        if not results or not results.get("documents"):
            return contexts

        for i in range(len(results["documents"][0])):
            metadata = results["metadatas"][0][i]

            contexts.append(
                RetrievedContext(
                    text=results["documents"][0][i],
                    source=metadata.get("source", "manual_kb"),
                    relevance_score=1.0 - results["distances"][0][i],
                    metadata=metadata
                )
            )
        
        return contexts
    
    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(self.collection.name)


class MathKnowledgeBase(VectorStore):
    """Specialized vector store for math knowledge"""
    
    def __init__(self):
        super().__init__(collection_name="math_knowledge")
    
    def search_by_topic(
        self, 
        query: str, 
        topic: str, 
        k: int = 3
    ) -> list[RetrievedContext]:
        """Search within a specific topic"""
        return self.search(
            query,
            k=k,
            filter_metadata={"topic": topic}
        )
    
    def get_formulas(self, topic: str) -> list[RetrievedContext]:
        """Get formulas for a topic"""
        return self.search(
            f"{topic} formulas",
            k=5,
            filter_metadata={"type": "formula"}
        )
    
    def get_common_mistakes(self, topic: str) -> list[RetrievedContext]:
        """Get common mistakes for a topic"""
        return self.search(
            f"{topic} common mistakes pitfalls",
            k=3,
            filter_metadata={"type": "common_mistakes"}
        )
