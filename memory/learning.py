"""
Self-learning module that improves system performance based on feedback and patterns.
"""

from typing import Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json
import re

from memory.store import MemoryStore
from memory.vector_memory import VectorMemory
from models.schemas import MathTopic, UserFeedback, ParsedProblem
from config import Config


class LearningModule:
    """
    Implements self-learning capabilities:
    1. Pattern recognition from successful solutions
    2. OCR/ASR correction learning
    3. Solution strategy optimization
    4. Common mistake identification
    """
    
    def __init__(self, memory_store: MemoryStore, vector_memory: VectorMemory):
        self.memory = memory_store
        self.vector_memory = vector_memory
        self._ocr_correction_cache = {}
        self._solution_pattern_cache = {}
        self._load_learned_patterns()
    
    def _load_learned_patterns(self):
        """Load previously learned patterns into cache"""
        # Load OCR corrections
        corrections = self.memory.get_ocr_corrections()
        for original, corrected in corrections:
            self._ocr_correction_cache[original.lower().strip()] = corrected
        
        # Load successful solution patterns by topic
        for topic in MathTopic:
            if topic != MathTopic.UNKNOWN:
                patterns = self._extract_solution_patterns(topic.value)
                self._solution_pattern_cache[topic.value] = patterns
    
    def _extract_solution_patterns(self, topic: str, limit: int = 20) -> list[dict]:
        """Extract solution patterns from successful solutions"""
        successful = self.memory.get_successful_solutions(topic, limit=limit)
        
        patterns = []
        for sol in successful:
            pattern = {
                "problem_type": self._identify_problem_type(sol["problem"]),
                "solution_structure": self._extract_structure(sol["solution"]),
                "key_steps": self._extract_key_steps(sol["solution"]),
                "formulas_used": self._extract_formulas(sol["solution"])
            }
            patterns.append(pattern)
        
        return patterns
    
    def _identify_problem_type(self, problem: str) -> str:
        """Identify the type of problem from text"""
        problem_lower = problem.lower()
        
        type_keywords = {
            "solve_equation": ["solve", "find x", "find the value", "find the root"],
            "simplify": ["simplify", "reduce", "express"],
            "differentiate": ["differentiate", "derivative", "d/dx", "find dy/dx"],
            "integrate": ["integrate", "integral", "find the area"],
            "evaluate": ["evaluate", "calculate", "compute", "find the value of"],
            "prove": ["prove", "show that", "verify"],
            "find_probability": ["probability", "chance", "likelihood"],
            "find_limit": ["limit", "lim", "approaches"],
            "matrix_operation": ["matrix", "determinant", "inverse", "eigenvalue"]
        }
        
        for problem_type, keywords in type_keywords.items():
            if any(kw in problem_lower for kw in keywords):
                return problem_type
        
        return "general"
    
    def _extract_structure(self, solution: str) -> list[str]:
        """Extract the structural pattern of a solution"""
        structure = []
        
        # Identify common structural elements
        if re.search(r'given|let|assume', solution, re.IGNORECASE):
            structure.append("setup")
        if re.search(r'using|apply|by', solution, re.IGNORECASE):
            structure.append("method_application")
        if re.search(r'substitut|plug|replace', solution, re.IGNORECASE):
            structure.append("substitution")
        if re.search(r'simplif|reduc|cancel', solution, re.IGNORECASE):
            structure.append("simplification")
        if re.search(r'therefore|thus|hence|answer|result', solution, re.IGNORECASE):
            structure.append("conclusion")
        if re.search(r'verify|check|confirm', solution, re.IGNORECASE):
            structure.append("verification")
        
        return structure
    
    def _extract_key_steps(self, solution: str) -> list[str]:
        """Extract key steps from solution"""
        steps = []
        
        # Look for numbered steps
        step_matches = re.findall(r'(?:step\s*)?(\d+)[.):]\s*(.+?)(?=(?:step\s*)?\d+[.)]|$)', 
                                   solution, re.IGNORECASE | re.DOTALL)
        
        if step_matches:
            for _, step_content in step_matches:
                # Summarize each step
                step_summary = step_content.strip()[:100]
                steps.append(step_summary)
        
        return steps
    
    def _extract_formulas(self, solution: str) -> list[str]:
        """Extract mathematical formulas used"""
        formulas = []
        
        # Common formula patterns
        formula_patterns = [
            r'x\s*=\s*\([^)]+\)',  # x = (...)
            r'\\frac\{[^}]+\}\{[^}]+\}',  # LaTeX fractions
            r'\w+\s*=\s*[-+*/\d\w\s\^]+',  # Variable assignments
            r'(?:sin|cos|tan|log|ln|sqrt)\s*\([^)]+\)',  # Functions
        ]
        
        for pattern in formula_patterns:
            matches = re.findall(pattern, solution)
            formulas.extend(matches[:3])  # Limit to 3 per pattern
        
        return list(set(formulas))[:10]  # Deduplicate and limit
    
    def apply_ocr_corrections(self, text: str) -> tuple[str, list[str]]:
        """
        Apply learned OCR corrections to text.
        Returns corrected text and list of corrections made.
        """
        corrections_made = []
        corrected_text = text
        
        # Apply exact match corrections
        for original, corrected in self._ocr_correction_cache.items():
            if original in text.lower():
                corrected_text = re.sub(
                    re.escape(original), 
                    corrected, 
                    corrected_text, 
                    flags=re.IGNORECASE
                )
                corrections_made.append(f"'{original}' → '{corrected}'")
        
        # Apply common OCR error patterns
        common_ocr_errors = {
            r'\bl\b': '1',  # l -> 1 in math context
            r'\bO\b': '0',  # O -> 0 in math context
            r'\bx\s*x\b': 'x²',  # xx -> x²
            r'(\d)\s*x\s*(\d)': r'\1×\2',  # multiplication
        }
        
        for pattern, replacement in common_ocr_errors.items():
            if re.search(pattern, corrected_text):
                old_text = corrected_text
                corrected_text = re.sub(pattern, replacement, corrected_text)
                if old_text != corrected_text:
                    corrections_made.append(f"Pattern correction: {pattern}")
        
        return corrected_text, corrections_made
    
    def get_solution_hints(self, parsed_problem: ParsedProblem) -> dict:
        """
        Get hints for solving based on learned patterns.
        """
        topic = parsed_problem.topic.value
        problem_type = self._identify_problem_type(parsed_problem.problem_text)
        
        hints = {
            "recommended_approach": [],
            "similar_solutions": [],
            "common_pitfalls": [],
            "expected_structure": []
        }
        
        # Get patterns for this topic
        patterns = self._solution_pattern_cache.get(topic, [])
        
        # Find similar patterns
        for pattern in patterns:
            if pattern["problem_type"] == problem_type:
                hints["expected_structure"].extend(pattern["solution_structure"])
                hints["recommended_approach"].extend(pattern["key_steps"][:3])
        
        # Deduplicate
        hints["expected_structure"] = list(set(hints["expected_structure"]))
        hints["recommended_approach"] = list(set(hints["recommended_approach"]))[:5]
        
        # Get similar solved problems from vector memory
        similar = self.vector_memory.find_similar(
            parsed_problem.problem_text, 
            k=3,
            filter_feedback="correct"
        )
        hints["similar_solutions"] = similar
        
        # Get common pitfalls from feedback
        pitfalls = self._get_common_pitfalls(topic, problem_type)
        hints["common_pitfalls"] = pitfalls
        
        return hints
    
    def _get_common_pitfalls(self, topic: str, problem_type: str) -> list[str]:
        """Get common pitfalls based on incorrect feedback"""
        pitfalls = []
        
        # Query memory for incorrect solutions
        cursor = self.memory.conn.execute("""
            SELECT f.comment, f.corrected_solution
            FROM feedback_log f
            JOIN interactions i ON f.interaction_id = i.id
            WHERE i.topic = ? AND f.feedback_type IN ('incorrect', 'partial')
            ORDER BY f.timestamp DESC
            LIMIT 10
        """, (topic,))
        
        for comment, correction in cursor.fetchall():
            if comment:
                pitfalls.append(comment)
        
        return pitfalls[:5]
    
    def learn_from_feedback(self, feedback: UserFeedback, interaction_id: int):
        """
        Learn from user feedback to improve future performance.
        """
        # Get the interaction details
        cursor = self.memory.conn.execute("""
            SELECT topic, parsed_problem, solution
            FROM interactions
            WHERE id = ?
        """, (interaction_id,))
        
        row = cursor.fetchone()
        if not row:
            return
        
        topic, problem, solution = row
        
        if feedback.feedback_type == "correct":
            # Reinforce successful patterns
            self._reinforce_pattern(topic, problem, solution)
        
        elif feedback.feedback_type == "incorrect" and feedback.corrected_solution:
            # Learn from correction
            self._learn_from_correction(
                topic, 
                problem, 
                solution, 
                feedback.corrected_solution
            )
        
        elif feedback.feedback_type == "partial" and feedback.comment:
            # Record partial success notes
            self._record_improvement_note(topic, problem, feedback.comment)
        
        # Update vector memory with feedback
        self.vector_memory.update_feedback(interaction_id, feedback.feedback_type)
        
        # Refresh pattern cache for this topic
        self._solution_pattern_cache[topic] = self._extract_solution_patterns(topic)
    
    def _reinforce_pattern(self, topic: str, problem: str, solution: str):
        """Reinforce a successful solution pattern"""
        # Add to high-confidence solutions
        self.memory.conn.execute("""
            INSERT OR REPLACE INTO reinforced_patterns 
            (topic, problem_type, solution_pattern, success_count, last_success)
            VALUES (?, ?, ?, 
                COALESCE((SELECT success_count FROM reinforced_patterns 
                          WHERE topic = ? AND problem_type = ?), 0) + 1,
                ?)
        """, (
            topic,
            self._identify_problem_type(problem),
            json.dumps(self._extract_structure(solution)),
            topic,
            self._identify_problem_type(problem),
            datetime.now().isoformat()
        ))
        self.memory.conn.commit()
    
    def _learn_from_correction(
        self, 
        topic: str, 
        problem: str, 
        wrong_solution: str, 
        correct_solution: str
    ):
        """Learn from a correction"""
        # Store the correction for future reference
        self.memory.conn.execute("""
            INSERT INTO solution_corrections 
            (topic, problem, wrong_solution, correct_solution, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (topic, problem, wrong_solution, correct_solution, datetime.now().isoformat()))
        self.memory.conn.commit()
        
        # Analyze what went wrong
        wrong_structure = self._extract_structure(wrong_solution)
        correct_structure = self._extract_structure(correct_solution)
        
        # Log the difference
        missing_steps = set(correct_structure) - set(wrong_structure)
        if missing_steps:
            self.memory.conn.execute("""
                INSERT INTO learning_notes 
                (topic, note_type, content, timestamp)
                VALUES (?, 'missing_steps', ?, ?)
            """, (topic, json.dumps(list(missing_steps)), datetime.now().isoformat()))
            self.memory.conn.commit()
    
    def _record_improvement_note(self, topic: str, problem: str, comment: str):
        """Record a note for improvement"""
        self.memory.conn.execute("""
            INSERT INTO learning_notes 
            (topic, note_type, content, timestamp)
            VALUES (?, 'partial_feedback', ?, ?)
        """, (topic, comment, datetime.now().isoformat()))
        self.memory.conn.commit()
    
    def get_learning_stats(self) -> dict:
        """Get statistics about learning progress"""
        stats = {
            "ocr_corrections_learned": len(self._ocr_correction_cache),
            "solution_patterns_by_topic": {},
            "feedback_distribution": self.memory.get_feedback_stats(),
            "improvement_over_time": []
        }
        
        # Count patterns per topic
        for topic, patterns in self._solution_pattern_cache.items():
            stats["solution_patterns_by_topic"][topic] = len(patterns)
        
        # Calculate improvement over time (last 7 days)
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            cursor = self.memory.conn.execute("""
                SELECT 
                    COUNT(CASE WHEN user_feedback = 'correct' THEN 1 END) as correct,
                    COUNT(CASE WHEN user_feedback = 'incorrect' THEN 1 END) as incorrect
                FROM interactions
                WHERE DATE(timestamp) = ?
            """, (date,))
            row = cursor.fetchone()
            if row[0] + row[1] > 0:
                accuracy = row[0] / (row[0] + row[1])
            else:
                accuracy = None
            stats["improvement_over_time"].append({
                "date": date,
                "correct": row[0],
                "incorrect": row[1],
                "accuracy": accuracy
            })
        
        return stats
    
    def suggest_knowledge_base_updates(self) -> list[dict]:
        """
        Analyze feedback to suggest knowledge base updates.
        """
        suggestions = []
        
        # Find topics with high incorrect rate
        cursor = self.memory.conn.execute("""
            SELECT topic, 
                   COUNT(CASE WHEN user_feedback = 'correct' THEN 1 END) as correct,
                   COUNT(CASE WHEN user_feedback = 'incorrect' THEN 1 END) as incorrect
            FROM interactions
            WHERE user_feedback IS NOT NULL
            GROUP BY topic
            HAVING incorrect > 2
        """)
        
        for topic, correct, incorrect in cursor.fetchall():
            if incorrect > correct:
                suggestions.append({
                    "type": "knowledge_gap",
                    "topic": topic,
                    "message": f"High error rate in {topic} ({incorrect} incorrect vs {correct} correct). Consider adding more formulas/examples.",
                    "priority": "high"
                })
        
        # Find recurring feedback comments
        cursor = self.memory.conn.execute("""
            SELECT comment, COUNT(*) as count
            FROM feedback_log
            WHERE comment IS NOT NULL AND comment != ''
            GROUP BY comment
            HAVING count > 1
            ORDER BY count DESC
            LIMIT 5
        """)
        
        for comment, count in cursor.fetchall():
            suggestions.append({
                "type": "recurring_issue",
                "message": f"Recurring feedback ({count}x): {comment}",
                "priority": "medium"
            })
        
        return suggestions


class AdaptiveLearner:
    """
    Advanced learner that adapts solving strategies based on patterns.
    """
    
    def __init__(self, learning_module: LearningModule):
        self.learner = learning_module
        self.strategy_weights = defaultdict(lambda: defaultdict(float))
        self._load_strategy_weights()
    
    def _load_strategy_weights(self):
        """Load strategy success weights from history"""
        cursor = self.learner.memory.conn.execute("""
            SELECT topic, 
                   json_extract(parsed_problem, '$.question_type') as q_type,
                   user_feedback
            FROM interactions
            WHERE user_feedback IS NOT NULL
        """)
        
        for topic, q_type, feedback in cursor.fetchall():
            if q_type:
                weight_delta = 1.0 if feedback == "correct" else -0.5
                self.strategy_weights[topic][q_type] += weight_delta
    
    def get_best_strategy(self, topic: str, question_type: str) -> dict:
        """Get the best solving strategy based on historical success"""
        strategies = {
            "solve_equation": {
                "primary": "algebraic_manipulation",
                "tools": ["solve_equation", "simplify_expression"],
                "steps": ["identify equation type", "isolate variable", "solve", "verify"]
            },
            "differentiate": {
                "primary": "formula_application",
                "tools": ["differentiate", "simplify_expression"],
                "steps": ["identify function type", "apply differentiation rules", "simplify"]
            },
            "integrate": {
                "primary": "formula_application", 
                "tools": ["integrate", "simplify_expression"],
                "steps": ["identify integral type", "apply integration technique", "add constant"]
            },
            "find_probability": {
                "primary": "step_by_step_derivation",
                "tools": ["calculate"],
                "steps": ["identify sample space", "count favorable outcomes", "apply formula"]
            },
            "find_limit": {
                "primary": "formula_application",
                "tools": ["evaluate_limit", "simplify_expression"],
                "steps": ["check direct substitution", "apply L'Hopital if needed", "evaluate"]
            },
            "matrix_operation": {
                "primary": "direct_computation",
                "tools": ["calculate"],
                "steps": ["identify operation", "apply matrix rules", "compute"]
            },
            "general": {
                "primary": "step_by_step_derivation",
                "tools": ["calculate", "simplify_expression"],
                "steps": ["understand problem", "identify method", "solve", "verify"]
            }
        }
        
        base_strategy = strategies.get(question_type, strategies["general"])
        
        # Adjust based on historical success
        weight = self.strategy_weights[topic].get(question_type, 0)
        
        if weight < -2:
            # This strategy often fails, suggest more verification
            base_strategy["steps"].append("double-check each step")
            base_strategy["confidence_adjustment"] = -0.1
        elif weight > 5:
            # This strategy works well
            base_strategy["confidence_adjustment"] = 0.1
        else:
            base_strategy["confidence_adjustment"] = 0
        
        return base_strategy
    
    def record_strategy_outcome(self, topic: str, question_type: str, success: bool):
        """Record the outcome of using a strategy"""
        weight_delta = 1.0 if success else -0.5
        self.strategy_weights[topic][question_type] += weight_delta