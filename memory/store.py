import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
from models.schemas import MemoryEntry, MathTopic, InputType, UserFeedback
from rag.embeddings import EmbeddingService
import chromadb
from config import Config
import json

class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        
        # Vector memory for similarity search
        self.embedding_service = EmbeddingService()
        self.chroma_client = chromadb.PersistentClient(
            path=str(Path(Config.CHROMA_PERSIST_DIR) / "memory")
        )
        self.memory_collection = self.chroma_client.get_or_create_collection("problem_memory")
    
    def _init_db(self):
        """Initialize database tables"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_type TEXT NOT NULL,
                raw_input TEXT NOT NULL,
                parsed_problem TEXT NOT NULL,
                topic TEXT NOT NULL,
                solution TEXT NOT NULL,
                verification_score REAL,
                user_feedback TEXT,
                corrected_solution TEXT,
                embedding_id TEXT
            );
            
            CREATE TABLE IF NOT EXISTS feedback_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER,
                feedback_type TEXT NOT NULL,
                comment TEXT,
                corrected_solution TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (interaction_id) REFERENCES interactions(id)
            );
            
            CREATE TABLE IF NOT EXISTS ocr_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_text TEXT NOT NULL,
                corrected_text TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            
            CREATE INDEX IF NOT EXISTS idx_topic ON interactions(topic);
            CREATE INDEX IF NOT EXISTS idx_feedback ON interactions(user_feedback);
        """)
        self.conn.commit()
    
    def save_interaction(self, entry: MemoryEntry) -> int:
        """Save an interaction to memory"""
        cursor = self.conn.execute("""
            INSERT INTO interactions 
            (timestamp, input_type, raw_input, parsed_problem, topic, solution, verification_score, user_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.timestamp.isoformat(),
            entry.input_type.value,
            entry.raw_input,
            entry.parsed_problem,
            entry.topic.value,
            entry.solution,
            entry.verification_score,
            entry.user_feedback
        ))
        self.conn.commit()
        
        interaction_id = cursor.lastrowid
        
        # Add to vector memory
        embedding_id = f"interaction_{interaction_id}"
        embedding = self.embedding_service.embed_single(entry.parsed_problem)
        
        self.memory_collection.add(
            ids=[embedding_id],
            embeddings=[embedding],
            documents=[entry.parsed_problem],
            metadatas=[{
                "interaction_id": interaction_id,
                "topic": entry.topic.value,
                "solution": entry.solution[:1000],
                "feedback": entry.user_feedback or ""
            }]
        )
        
        # Update embedding_id in SQL
        self.conn.execute(
            "UPDATE interactions SET embedding_id = ? WHERE id = ?",
            (embedding_id, interaction_id)
        )
        self.conn.commit()
        
        return interaction_id
    
    def save_feedback(self, interaction_id: int, feedback: UserFeedback):
        """Save user feedback"""
        self.conn.execute("""
            INSERT INTO feedback_log 
            (interaction_id, feedback_type, comment, corrected_solution, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            interaction_id,
            feedback.feedback_type,
            feedback.comment,
            feedback.corrected_solution,
            feedback.timestamp.isoformat()
        ))
        
        # Update interaction record
        self.conn.execute("""
            UPDATE interactions 
            SET user_feedback = ?, corrected_solution = ?
            WHERE id = ?
        """, (feedback.feedback_type, feedback.corrected_solution, interaction_id))
        
        self.conn.commit()
    
    def save_ocr_correction(self, original: str, corrected: str):
        """Save OCR correction for learning"""
        self.conn.execute("""
            INSERT INTO ocr_corrections (original_text, corrected_text, timestamp)
            VALUES (?, ?, ?)
        """, (original, corrected, datetime.now().isoformat()))
        self.conn.commit()
    
    def find_similar_problems(self, problem_text: str, k: int = 3) -> list[dict]:
        """Find similar problems from memory"""
        embedding = self.embedding_service.embed_single(problem_text)
        
        results = self.memory_collection.query(
            query_embeddings=[embedding],
            n_results=k
        )
        
        similar = []
        for i in range(len(results["documents"][0])):
            similar.append({
                "problem": results["documents"][0][i],
                "solution": results["metadatas"][0][i].get("solution", ""),
                "topic": results["metadatas"][0][i].get("topic", ""),
                "feedback": results["metadatas"][0][i].get("feedback", ""),
                "similarity": 1 - results["distances"][0][i]
            })
        
        return similar
    
    def get_successful_solutions(self, topic: str, limit: int = 5) -> list[dict]:
        """Get successful solutions for a topic"""
        cursor = self.conn.execute("""
            SELECT raw_input, parsed_problem, solution 
            FROM interactions 
            WHERE topic = ? AND user_feedback = 'correct'
            ORDER BY timestamp DESC
            LIMIT ?
        """, (topic, limit))
        
        return [
            {"raw": row[0], "problem": row[1], "solution": row[2]}
            for row in cursor.fetchall()
        ]
    
    def get_ocr_corrections(self) -> list[tuple[str, str]]:
        """Get all OCR corrections for pattern learning"""
        cursor = self.conn.execute(
            "SELECT original_text, corrected_text FROM ocr_corrections"
        )
        return cursor.fetchall()
    
    def get_feedback_stats(self) -> dict:
        """Get feedback statistics"""
        cursor = self.conn.execute("""
            SELECT user_feedback, COUNT(*) 
            FROM interactions 
            WHERE user_feedback IS NOT NULL AND user_feedback != ''
            GROUP BY user_feedback
        """)
        return dict(cursor.fetchall())
    
    def get_recent_interactions(self, limit: int = 10) -> list[dict]:
        """Get recent interactions"""
        cursor = self.conn.execute("""
            SELECT id, timestamp, topic, parsed_problem, solution, user_feedback
            FROM interactions
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "topic": row[2],
                "problem": row[3],
                "solution": row[4],
                "feedback": row[5]
            }
            for row in cursor.fetchall()
        ]