from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
from enum import Enum

class InputType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"

class MathTopic(str, Enum):
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    PROBABILITY = "probability"
    LINEAR_ALGEBRA = "linear_algebra"
    UNKNOWN = "unknown"

class RawInput(BaseModel):
    """Raw input from user"""
    type: InputType
    content: str  # text, file path, or processed text
    original_content: Optional[str] = None  # original file path for image/audio
    confidence: float = 1.0
    needs_review: bool = False
    metadata: dict = Field(default_factory=dict)

class ParsedProblem(BaseModel):
    """Structured math problem"""
    problem_text: str
    topic: MathTopic = MathTopic.UNKNOWN
    subtopic: Optional[str] = None
    variables: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    given_values: dict = Field(default_factory=dict)
    question_type: Optional[str] = None  # "solve", "prove", "find", "evaluate"
    needs_clarification: bool = False
    clarification_reason: Optional[str] = None
    confidence: float = 1.0

class RetrievedContext(BaseModel):
    """Retrieved knowledge from RAG"""
    text: str
    source: str
    relevance_score: float
    metadata: dict = Field(default_factory=dict)

class SolutionStep(BaseModel):
    """Single step in solution"""
    step_number: int
    description: str
    mathematical_expression: Optional[str] = None
    explanation: Optional[str] = None

class Solution(BaseModel):
    """Complete solution"""
    final_answer: str
    steps: list[SolutionStep] = Field(default_factory=list)
    method_used: str
    context_used: list[RetrievedContext] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)
    confidence: float = 1.0

class Verification(BaseModel):
    """Verification result"""
    is_correct: bool
    confidence: float
    issues_found: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    edge_cases_checked: list[str] = Field(default_factory=list)

class Explanation(BaseModel):
    """Student-friendly explanation"""
    summary: str
    detailed_steps: list[str] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    common_mistakes_to_avoid: list[str] = Field(default_factory=list)
    related_problems: list[str] = Field(default_factory=list)

class AgentTrace(BaseModel):
    """Trace of agent execution"""
    agent_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    input_summary: str
    output_summary: Optional[str] = None
    status: Literal["running", "completed", "failed", "needs_hitl"] = "running"
    error: Optional[str] = None

class FinalResult(BaseModel):
    """Complete result from the system"""
    status: Literal["success", "needs_hitl", "error"]
    raw_input: RawInput
    parsed_problem: Optional[ParsedProblem] = None
    solution: Optional[Solution] = None
    verification: Optional[Verification] = None
    explanation: Optional[Explanation] = None
    agent_traces: list[AgentTrace] = Field(default_factory=list)
    hitl_reason: Optional[str] = None
    error_message: Optional[str] = None

class UserFeedback(BaseModel):
    """User feedback on solution"""
    feedback_type: Literal["correct", "partial", "incorrect"]
    comment: Optional[str] = None
    corrected_solution: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class MemoryEntry(BaseModel):
    """Entry in memory store"""
    id: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    input_type: InputType
    raw_input: str
    parsed_problem: str
    topic: MathTopic
    solution: str
    verification_score: float
    user_feedback: Optional[str] = None
    corrected_solution: Optional[str] = None
    embedding_id: Optional[str] = None