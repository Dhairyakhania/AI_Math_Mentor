"""
Math Mentor Team using Agno framework.
"""

from agents.leader_agent import LeaderAgent
from models.schemas import RawInput, FinalResult
from config import Config


class MathMentorTeam:
    """
    Main team class for the Math Mentor application.
    Acts as the single public entry point for problem solving.
    """

    def __init__(self):
        # Fail fast on misconfiguration
        if Config.LLM_PROVIDER == "groq" and not Config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is required for Groq provider")

        if Config.LLM_PROVIDER == "gemini" and not Config.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini provider")

        self.leader = LeaderAgent()

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def solve(self, raw_input: RawInput) -> FinalResult:
        """
        Solve a math problem end-to-end.
        This is the ONLY method app.py should call.
        """
        if not isinstance(raw_input, RawInput):
            raise TypeError(
                f"MathMentorTeam.solve expected RawInput, "
                f"got {type(raw_input).__name__}"
            )

        return self.leader.solve(raw_input)

    # -------------------------------------------------
    # Metadata / UI helpers
    # -------------------------------------------------

    def get_team_info(self) -> dict:
        """Return static metadata about the team."""
        model_name = (
            Config.GEMINI_MODEL
            if Config.LLM_PROVIDER == "gemini"
            else Config.GROQ_LLM_MODEL
        )

        return {
            "name": "MathMentorTeam",
            "framework": "Agno",
            "llm_provider": Config.LLM_PROVIDER,
            "llm_model": model_name,
            "pipeline": [
                "Parser",
                "Strategy",
                "Router",
                "Solver",
                "Verifier",
                "Explainer",
            ],
            "agents": [
                {"name": "Parser", "role": "Parse raw input into structured problem"},
                {"name": "Strategy", "role": "Plan solution approach"},
                {"name": "Router", "role": "Select solving strategy"},
                {"name": "Solver", "role": "Solve problem using tools"},
                {"name": "Verifier", "role": "Verify correctness and confidence"},
                {"name": "Explainer", "role": "Generate student-friendly explanation"},
            ],
        }
