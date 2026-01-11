"""
Knowledge Base Ingestion Script
-------------------------------
Reads markdown files with YAML headers,
chunks content, embeds it, and stores it in ChromaDB.

Run once after updating knowledge_base/.
"""

import yaml
from pathlib import Path
from rag.vectorstore import MathKnowledgeBase
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------
# CONFIG
# -------------------------------

PROJECT_ROOT = Path(__file__).resolve()
RAG_DIR = PROJECT_ROOT.parent
KB_DIR = RAG_DIR / "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def ingest():
    if not KB_DIR.exists():
        raise FileNotFoundError("knowledge_base directory not found")

    kb = MathKnowledgeBase()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    documents = []

    for file in KB_DIR.glob("*.md"):
        raw_text = file.read_text(encoding="utf-8").strip()

        # -------------------------------
        # YAML METADATA PARSING
        # -------------------------------
        if not raw_text.startswith("---"):
            raise ValueError(f"Missing YAML header in {file.name}")

        try:
            _, yaml_block, content = raw_text.split("---", 2)
            metadata = yaml.safe_load(yaml_block)
        except Exception as e:
            raise ValueError(f"Invalid YAML in {file.name}: {e}")

        # -------------------------------
        # METADATA VALIDATION
        # -------------------------------
        required_fields = {"topic", "type", "source"}
        if not required_fields.issubset(metadata):
            raise ValueError(
                f"{file.name} missing required metadata fields: {required_fields}"
            )

        # -------------------------------
        # CHUNK CONTENT
        # -------------------------------
        content = content.strip()
        chunks = splitter.split_text(content)

        for chunk in chunks:
            documents.append({
                "text": chunk,
                "metadata": {
                    "source": metadata["source"],
                    "topic": metadata["topic"],
                    "type": metadata["type"]
                }
            })

    if not documents:
        raise RuntimeError("No documents found for ingestion")

    kb.add_documents(documents)

    print(f"✅ Ingested {len(documents)} chunks into vector store.")

def ingest_kb():
    ingest()

if __name__ == "__main__":
    ingest_kb()
