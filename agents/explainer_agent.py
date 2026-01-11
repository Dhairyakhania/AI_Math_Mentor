"""
Explainer Agent using Agno framework.
Preserves equations and explains reasoning.
"""

from agents.base_agent import BaseAgent
from models.schemas import ParsedProblem, Solution, Verification, Explanation
from typing import Any
import re


class ExplainerAgent(BaseAgent):
    """
    Equation-first explainer.
    Cleans solver metadata and presents math clearly.
    """

    def __init__(self):
        super().__init__(
            name="Explainer",
            description="Explains solver steps with equations and reasoning",
            instructions=[
                "You are a JEE-level math tutor.",
                "ALWAYS show equations if present.",
                "Explain WHY each step is valid.",
                "Do NOT redo calculations.",
                "Do NOT add new steps.",
                "Do NOT expose solver metadata.",
                "Output ONLY valid JSON."
            ],
            tools=[]
        )

    def process(self, input_data: Any) -> Explanation:
        if not (isinstance(input_data, tuple) and len(input_data) == 3):
            raise ValueError("Expected (ParsedProblem, Solution, Verification)")
        return self.explain(*input_data)

    # -------------------------------------------------
    # Core explanation logic
    # -------------------------------------------------
    def explain(
        self,
        parsed_problem: ParsedProblem,
        solution: Solution,
        verification: Verification
    ) -> Explanation:

        detailed_steps = [
            self._format_step(step.description, step.step_number)
            for step in solution.steps
        ]

        summary = f"The problem was solved step by step to obtain {solution.final_answer}."

        return Explanation(
            summary=summary,
            detailed_steps=detailed_steps,
            key_concepts=self._infer_key_concepts(parsed_problem),
            common_mistakes_to_avoid=self._infer_common_mistakes(parsed_problem),
            related_problems=[parsed_problem.topic.value],
            encouragement="Keep practicing — understanding transformations builds confidence."
        )

    # -------------------------------------------------
    # Step formatter (UPDATED FOR INTEGRATION)
    # -------------------------------------------------
    def _format_step(self, text: str, step_number: int) -> str:
        """
        Converts solver output into:
        Step N:
        <equations>
        Explanation: <reasoning>
        """

        # Remove solver labels
        cleaned = re.sub(
            r"(STEP\s*\d+:|ACTION:|RESULT:|PRINCIPLE:|FINAL ANSWER:)",
            "",
            text,
            flags=re.IGNORECASE
        )

        lines = [l.strip() for l in cleaned.splitlines() if l.strip()]

        # Extract equations (DIFFERENTIATION + INTEGRATION + PROBABILITY)
        equations = [
            l for l in lines
            if (
                "=" in l
                or "→" in l
                or "d/dx" in l
                or "∫" in l
                or "integral" in l.lower()
                or "+ c" in l.lower()
                or "+c" in l.lower()
                or re.search(r"\bP\(", l)
            )
        ]

        raw_reasoning = " ".join(
            l for l in lines if l not in equations
        )

        explanation = self._normalize_reasoning(raw_reasoning)

        output = f"Step {step_number}:\n"

        if equations:
            output += "\n".join(equations)

        if explanation:
            output += "\nExplanation: " + explanation

        return output.strip()

    # -------------------------------------------------
    # Reasoning normalization (DIFF + INTEGRATION)
    # -------------------------------------------------
    def _normalize_reasoning(self, text: str) -> str:
        """
        Converts solver-style phrases into student explanations.
        Generalised across calculus topics.
        """

        text = text.lower()

        replacements = {
            # Differentiation
            "power rule of differentiation": (
                "The power rule is applied, which states that "
                "d/dx(xⁿ) = n·xⁿ⁻¹."
            ),
            "apply the power rule to each term": (
                "Each term is differentiated independently using the power rule."
            ),
            "differentiate the function": (
                "We compute the derivative of the function with respect to x."
            ),

            # Integration
            "apply integration": (
                "The integral is evaluated using standard integration rules."
            ),
            "power rule of integration": (
                "The power rule for integration is applied, which increases the power by one "
                "and divides by the new exponent."
            ),
            "use substitution": (
                "A substitution is used to simplify the integrand before integrating."
            ),
            "integration by parts": (
                "Integration by parts is used, based on the formula "
                "∫u dv = uv − ∫v du."
            ),
            "constant of integration": (
                "A constant of integration is added because the derivative of a constant is zero."
            ),
            "+ c": (
                "A constant of integration is included to represent all antiderivatives."
            ),

            # Simplification
            "simplify the expression": (
                "The resulting terms are combined and simplified."
            ),
            "simplification": (
                "The expression is simplified by combining like terms."
            ),
        }

        for key, value in replacements.items():
            if key in text:
                return value

        return text.capitalize() if text else ""

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def _infer_key_concepts(self, parsed_problem: ParsedProblem) -> list[str]:
        topic = parsed_problem.topic.value
        if topic == "probability":
            return ["Probability = Favorable outcomes / Total outcomes"]
        if topic in ("algebra", "linear_algebra"):
            return ["Equation solving", "Algebraic manipulation"]
        if topic == "calculus":
            return ["Differentiation", "Integration"]
        return [topic]

    def _infer_common_mistakes(self, parsed_problem: ParsedProblem) -> list[str]:
        topic = parsed_problem.topic.value
        if topic == "probability":
            return ["Incorrect total outcomes", "Ignoring equally likely cases"]
        if topic == "calculus":
            return ["Forgetting +C in integration", "Incorrect rule application"]
        if topic == "algebra":
            return ["Sign errors", "Skipping steps"]
        return []
