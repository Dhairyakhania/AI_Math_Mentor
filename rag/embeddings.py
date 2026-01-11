"""
Free local embeddings using sentence-transformers.
No API costs - runs entirely locally.
"""

from sentence_transformers import SentenceTransformer
from typing import Union
from config import Config
import numpy as np


class EmbeddingService:
    """
    Local embedding service using sentence-transformers.
    Free and runs offline after initial model download.
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if EmbeddingService._model is None:
            print(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
            EmbeddingService._model = SentenceTransformer(Config.EMBEDDING_MODEL)
            print("Embedding model loaded successfully!")
    
    @property
    def model(self):
        return EmbeddingService._model
    
    def embed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
        
        Returns:
            List of embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Convert to list of lists for ChromaDB compatibility
        return embeddings.tolist()
    
    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for single text"""
        return self.embed(text)[0]
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        emb1 = np.array(self.embed_single(text1))
        emb2 = np.array(self.embed_single(text2))
        
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


# Alternative: Use Google's free embedding API (if you prefer cloud)
class GoogleEmbeddingService:
    """
    Google's embedding API (has free tier).
    Use this if you prefer cloud-based embeddings.
    """
    
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        self.model = "models/embedding-001"
    
    def embed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        """Generate embeddings using Google's API"""
        import google.generativeai as genai
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        
        return embeddings
    
    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for single text"""
        return self.embed(text)[0]