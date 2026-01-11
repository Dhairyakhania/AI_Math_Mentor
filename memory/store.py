import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
from models.schemas import (
    MemoryEntry,
    MathTopic,
    InputType,
    UserFeedback,
    RawInput,
    ParsedProblem,
    Solution,
    Verification,
)
from rag.embeddings import EmbeddingService
import chromadb
from config import Config


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
        self.memory_collection = self.chroma_client.get_or_create_collection(
            "problem_memory"
        )

    # -------------------------------------------------
    # DB INIT
    # -------------------------------------------------

    def _init_db(self):
        self.conn.executescript(
            """
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
            """
        )
        self.conn.commit()

    # =================================================
    # ✅ SAFE INPUT TYPE EXTRACTION (NEW)
    # =================================================

    def _get_input_type(self, raw_input: RawInput) -> InputType:
        """
        Safely extract InputType from RawInput without assuming schema.
        """
        # Try common attribute names
        value = (
            getattr(raw_input, "input_type", None)
            or getattr(raw_input, "type", None)
            or getattr(raw_input, "mode", None)
        )

        if isinstance(value, InputType):
            return value

        if isinstance(value, str):
            try:
                return InputType(value)
            except Exception:
                pass

        # Fallback (safe default)
        return InputType.TEXT

    # =================================================
    # PUBLIC API (USED BY app.py)
    # =================================================

    def store_success(
        self,
        raw_input: RawInput,
        parsed_problem: ParsedProblem,
        solution: Solution,
        verification: Verification,
    ):
        entry = MemoryEntry(
            timestamp=datetime.utcnow(),
            input_type=self._get_input_type(raw_input),
            raw_input=raw_input.content,
            parsed_problem=parsed_problem.problem_text,
            topic=parsed_problem.topic,
            solution=solution.final_answer,
            verification_score=verification.confidence,
            user_feedback="correct",
        )

        self.save_interaction(entry)

    def store_failure(
        self,
        raw_input: RawInput,
        parsed_problem: ParsedProblem,
        solution: Solution,
        correction: str,
    ):
        entry = MemoryEntry(
            timestamp=datetime.utcnow(),
            input_type=self._get_input_type(raw_input),
            raw_input=raw_input.content,
            parsed_problem=parsed_problem.problem_text,
            topic=parsed_problem.topic,
            solution=solution.final_answer,
            verification_score=None,
            user_feedback="incorrect",
        )

        interaction_id = self.save_interaction(entry)

        feedback = UserFeedback(
            interaction_id=interaction_id,
            feedback_type="incorrect",
            comment="User provided correction",
            corrected_solution=correction,
            timestamp=datetime.utcnow(),
        )

        self.save_feedback(interaction_id, feedback)

    # =================================================
    # CORE STORAGE (UNCHANGED)
    # =================================================

    def save_interaction(self, entry: MemoryEntry) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO interactions
            (timestamp, input_type, raw_input, parsed_problem, topic,
             solution, verification_score, user_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.timestamp.isoformat(),
                entry.input_type.value,
                entry.raw_input,
                entry.parsed_problem,
                entry.topic.value,
                entry.solution,
                entry.verification_score,
                entry.user_feedback,
            ),
        )
        self.conn.commit()

        interaction_id = cursor.lastrowid

        embedding_id = f"interaction_{interaction_id}"
        embedding = self.embedding_service.embed_single(entry.parsed_problem)

        self.memory_collection.add(
            ids=[embedding_id],
            embeddings=[embedding],
            documents=[entry.parsed_problem],
            metadatas=[
                {
                    "interaction_id": interaction_id,
                    "topic": entry.topic.value,
                    "solution": entry.solution[:1000],
                    "feedback": entry.user_feedback or "",
                }
            ],
        )

        self.conn.execute(
            "UPDATE interactions SET embedding_id = ? WHERE id = ?",
            (embedding_id, interaction_id),
        )
        self.conn.commit()

        return interaction_id

    def save_feedback(self, interaction_id: int, feedback: UserFeedback):
        self.conn.execute(
            """
            INSERT INTO feedback_log
            (interaction_id, feedback_type, comment,
             corrected_solution, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                interaction_id,
                feedback.feedback_type,
                feedback.comment,
                feedback.corrected_solution,
                feedback.timestamp.isoformat(),
            ),
        )

        self.conn.execute(
            """
            UPDATE interactions
            SET user_feedback = ?, corrected_solution = ?
            WHERE id = ?
            """,
            (
                feedback.feedback_type,
                feedback.corrected_solution,
                interaction_id,
            ),
        )

        self.conn.commit()

    # =================================================
    # RETRIEVAL
    # =================================================

    def find_similar_problems(self, problem_text: str, k: int = 3) -> list[dict]:
        embedding = self.embedding_service.embed_single(problem_text)

        results = self.memory_collection.query(
            query_embeddings=[embedding], n_results=k
        )

        similar = []
        for i in range(len(results["documents"][0])):
            similar.append(
                {
                    "problem": results["documents"][0][i],
                    "solution": results["metadatas"][0][i].get("solution", ""),
                    "topic": results["metadatas"][0][i].get("topic", ""),
                    "feedback": results["metadatas"][0][i].get("feedback", ""),
                    "similarity": 1 - results["distances"][0][i],
                }
            )

        return similar

    def get_ocr_corrections(self) -> list[tuple[str, str]]:
        cursor = self.conn.execute(
            "SELECT original_text, corrected_text FROM ocr_corrections"
        )
        return cursor.fetchall()
