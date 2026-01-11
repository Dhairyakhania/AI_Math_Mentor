import os
import re
from pathlib import Path
from rag.vectorstore import MathKnowledgeBase

def parse_markdown_with_frontmatter(content: str) -> list[dict]:
    """Parse markdown file with YAML frontmatter into chunks"""
    chunks = []
    
    # Split by frontmatter sections
    sections = re.split(r'^---\s*$', content, flags=re.MULTILINE)
    
    current_metadata = {}
    
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        
        # Check if this is frontmatter (key: value pairs)
        if re.match(r'^\w+:', section):
            # Parse frontmatter
            for line in section.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    current_metadata[key.strip()] = value.strip()
        else:
            # This is content - split into logical chunks
            content_chunks = split_content(section)
            for chunk in content_chunks:
                if chunk.strip():
                    chunks.append({
                        "text": chunk.strip(),
                        "metadata": current_metadata.copy()
                    })
    
    return chunks

def split_content(content: str, max_chunk_size: int = 500) -> list[str]:
    """Split content into chunks by headers or size"""
    chunks = []
    
    # Split by headers
    header_pattern = r'^#+\s+'
    parts = re.split(header_pattern, content, flags=re.MULTILINE)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        if len(part) <= max_chunk_size:
            chunks.append(part)
        else:
            # Split by paragraphs
            paragraphs = part.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) <= max_chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
            
            if current_chunk:
                chunks.append(current_chunk.strip())
    
    return chunks

def load_knowledge_base():
    """Load all knowledge base documents into vector store"""
    kb_dir = Path(__file__).parent.parent / "rag" / "knowledge_base"
    kb = MathKnowledgeBase()
    
    all_docs = []
    
    for md_file in kb_dir.glob("*.md"):
        print(f"Processing {md_file.name}...")
        
        with open(md_file, 'r') as f:
            content = f.read()
        
        chunks = parse_markdown_with_frontmatter(content)
        
        for chunk in chunks:
            chunk["metadata"]["source"] = md_file.name
            all_docs.append(chunk)
    
    print(f"Loading {len(all_docs)} chunks into vector store...")
    ids = kb.add_documents(all_docs)
    print(f"Successfully loaded {len(ids)} documents!")
    
    return ids

if __name__ == "__main__":
    load_knowledge_base()