import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    KNOWLEDGE_BASE_DIR = BASE_DIR / "rag" / "knowledge_base"
    
    # Google Gemini (via Agno)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    
    # Groq (via Agno) - for both LLM and Whisper
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.1-70b-versatile")
    GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")
    
    # Choose primary LLM provider: "gemini" or "groq"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
    
    # Embeddings (local, free)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Vector Store
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    
    # Memory
    MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "./data/memory.db")
    
    # Thresholds
    OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.75"))
    AUDIO_CONFIDENCE_THRESHOLD = float(os.getenv("AUDIO_CONFIDENCE_THRESHOLD", "0.70"))
    VERIFIER_CONFIDENCE_THRESHOLD = float(os.getenv("VERIFIER_CONFIDENCE_THRESHOLD", "0.80"))
    
    @classmethod
    def ensure_dirs(cls):
        cls.DATA_DIR.mkdir(exist_ok=True)
        Path(cls.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_llm_model(cls):
        """Get the appropriate Agno model based on provider"""
        if cls.LLM_PROVIDER == "groq":
            from agno.models.groq import Groq
            return Groq(id=cls.GROQ_LLM_MODEL, api_key=cls.GROQ_API_KEY)
        else:
            from agno.models.google import Gemini
            return Gemini(id=cls.GEMINI_MODEL, api_key=cls.GOOGLE_API_KEY)
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if cls.LLM_PROVIDER == "gemini" and not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY required for Gemini. Get it from https://makersuite.google.com/app/apikey")
        
        if cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY required for Groq. Get it from https://console.groq.com/keys")
        
        if not cls.GROQ_API_KEY:
            print("Warning: GROQ_API_KEY not set. Audio transcription will use fallback.")
        
        return True

Config.ensure_dirs()