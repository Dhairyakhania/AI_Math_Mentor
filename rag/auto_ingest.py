"""
Auto-ingestion helper.
Runs KB ingestion only if vector store is empty.
"""

import yaml
from pathlib import Path
from rag.vectorstore import MathKnowledgeBase
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.ingest_kb import ingest_kb

PROJECT_ROOT = Path(__file__).resolve()
RAG_DIR = PROJECT_ROOT.parent
KB_DIR = RAG_DIR / "knowledge_base"


def auto_ingest_if_needed():
    kb = MathKnowledgeBase()

    # Check if collection already has data
    try:
        count = kb.collection.count()
    except Exception:
        count = 0

    if count == 0:
        print("📚 Vector store empty. Running KB ingestion...")
        ingest_kb()
    else:
        print(f"📚 Vector store already initialized ({count} chunks). Skipping ingestion.")
