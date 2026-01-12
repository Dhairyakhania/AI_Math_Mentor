import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

# -------------------------------------------------
# SAFE ENV / STREAMLIT SECRETS LOADER
# -------------------------------------------------

try:
    import streamlit as st
    _SECRETS = st.secrets
except Exception:
    _SECRETS = {}


def get_config(key: str, default=None):
    """
    Read config from:
    1. Environment variables
    2. Streamlit secrets
    3. Default
    """
    return os.getenv(key) or _SECRETS.get(key, default)


class Config:
    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    KNOWLEDGE_BASE_DIR = BASE_DIR / "rag" / "knowledge_base"

    # -------------------------------------------------
    # LLM Provider
    # -------------------------------------------------
    LLM_PROVIDER = get_config("LLM_PROVIDER", "gemini")

    # -------------------------------------------------
    # Google Gemini
    # -------------------------------------------------
    GOOGLE_API_KEY = get_config("GOOGLE_API_KEY", "")
    GEMINI_MODEL = get_config("GEMINI_MODEL", "gemini-2.0-flash")

    # -------------------------------------------------
    # Groq (LLM + Whisper)
    # -------------------------------------------------
    GROQ_API_KEY = get_config("GROQ_API_KEY", "")
    GROQ_LLM_MODEL = get_config(
        "GROQ_LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
    )
    GROQ_WHISPER_MODEL = get_config(
        "WHISPER_MODEL", "whisper-large-v3"
    )

    # -------------------------------------------------
    # Embeddings
    # -------------------------------------------------
    EMBEDDING_MODEL = get_config(
        "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    # -------------------------------------------------
    # Vector Store
    # -------------------------------------------------
    CHROMA_PERSIST_DIR = get_config(
        "CHROMA_PERSIST_DIR", "./data/chroma_db"
    )

    # -------------------------------------------------
    # Memory
    # -------------------------------------------------
    MEMORY_DB_PATH = get_config(
        "MEMORY_DB_PATH", "./data/memory.db"
    )

    # -------------------------------------------------
    # Thresholds
    # -------------------------------------------------
    OCR_CONFIDENCE_THRESHOLD = float(
        get_config("OCR_CONFIDENCE_THRESHOLD", 0.75)
    )
    AUDIO_CONFIDENCE_THRESHOLD = float(
        get_config("AUDIO_CONFIDENCE_THRESHOLD", 0.70)
    )
    VERIFIER_CONFIDENCE_THRESHOLD = float(
        get_config("VERIFIER_CONFIDENCE_THRESHOLD", 0.80)
    )

    # -------------------------------------------------
    # Setup helpers
    # -------------------------------------------------
    @classmethod
    def ensure_dirs(cls):
        cls.DATA_DIR.mkdir(exist_ok=True)
        Path(cls.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_llm_model(cls):
        """Return the correct Agno LLM instance."""
        if cls.LLM_PROVIDER == "groq":
            from agno.models.groq import Groq
            return Groq(
                id=cls.GROQ_LLM_MODEL,
                api_key=cls.GROQ_API_KEY
            )
        else:
            from agno.models.google import Gemini
            return Gemini(
                id=cls.GEMINI_MODEL,
                api_key=cls.GOOGLE_API_KEY
            )

    @classmethod
    def validate(cls):
        """Validate configuration at startup."""
        if cls.LLM_PROVIDER == "gemini" and not cls.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY required for Gemini. "
                "Set it in Streamlit Secrets or environment variables."
            )

        if cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY required for Groq. "
                "Set it in Streamlit Secrets or environment variables."
            )

        if not cls.GROQ_API_KEY:
            print("Warning: GROQ_API_KEY not set. Audio transcription may fail.")

        return True


# Ensure directories on import
Config.ensure_dirs()
