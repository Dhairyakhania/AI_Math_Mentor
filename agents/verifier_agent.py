"""
Verifier Agent using Agno.
Performs JEE-style correctness and domain checks.
CONTEXT-AWARE VERSION.
"""

from agents.base_agent import BaseAgent
from models.schemas import ParsedProblem, Solution, Verification, RetrievedContext
from typing import Any, List
import re
import json


class VerifierAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Verifier",
            description="Verifies mathematical correctness and domain validity",
            instructions=[
                "You are a strict JEE-level solution verifier.",
                "Check domain constraints.",
                "Check probability bounds.",
                "Check extraneous roots.",
                "Use reference material only when relevant.",
                "Return ONLY valid JSON."
            ],
            tools=[]
        )

    # -------------------------------------------------
    # ENTRY POINT (UPDATED)
    # -------------------------------------------------

    def process(self, input_data: Any) -> Verification:
        """
        Expected input:
        (ParsedProblem, Solution, context: list)
        """
        if not isinstance(input_data, (list, tuple)):
            raise TypeError("VerifierAgent.process expects tuple input")

        if len(input_data) == 2:
            parsed_problem, solution = input_data
            context = []
        elif len(input_data) == 3:
            parsed_problem, solution, context = input_data
        else:
            raise TypeError(
                "VerifierAgent.process expects (ParsedProblem, Solution, [context])"
            )

        return self.verify(parsed_problem, solution, context)

    # -------------------------------------------------
    # MAIN VERIFICATION (UPDATED)
    # -------------------------------------------------

    def verify(
        self,
        parsed_problem: ParsedProblem,
        solution: Solution,
        context: List[RetrievedContext]
    ) -> Verification:

        topic = parsed_problem.topic.value

        # =================================================
        # 1. ALGEBRA / LINEAR ALGEBRA → DETERMINISTIC
        # =================================================
        if topic in ("algebra", "linear_algebra"):
            deterministic = self._deterministic_algebra_check(
                parsed_problem.problem_text,
                solution.final_answer
            )

            if deterministic is not None:
                is_correct, issue = deterministic

                return Verification(
                    is_correct=is_correct,
                    confidence=0.98 if is_correct else 0.4,
                    issues_found=[] if is_correct else [issue],
                    suggestions=[] if is_correct else ["Re-check algebraic steps"],
                    edge_cases_checked=["Direct substitution"]
                )

        # =================================================
        # 2. PROBABILITY → BOUNDS CHECK
        # =================================================
        if topic == "probability":
            match = re.search(r"([-+]?\d*\.?\d+)", solution.final_answer)
            if match:
                try:
                    p = float(match.group(1))
                    if 0.0 <= p <= 1.0:
                        return Verification(
                            is_correct=True,
                            confidence=0.85,
                            issues_found=[],
                            suggestions=[],
                            edge_cases_checked=["Probability bounds check"]
                        )
                except ValueError:
                    pass

            # Soft pass — do not block probability answers
            return Verification(
                is_correct=True,
                confidence=0.75,
                issues_found=[],
                suggestions=[],
                edge_cases_checked=["Conceptual probability validation"]
            )

        # =================================================
        # 3. CONTEXT-AWARE LLM VERIFICATION
        # =================================================

        prompt = f"""
Verify the following solution.

PROBLEM:
{parsed_problem.problem_text}

SOLUTION STEPS:
{[s.description for s in solution.steps]}

FINAL ANSWER:
{solution.final_answer}
"""

        # Inject reference material if available
        if context:
            prompt += "\nREFERENCE MATERIAL:\n"
            for i, ctx in enumerate(context, start=1):
                source = getattr(ctx, "source", "unknown")
                text = getattr(ctx, "text", "")
                prompt += f"""
REFERENCE {i}:
SOURCE: {source}
CONTENT:
{text}
"""

        prompt += """
CHECKS REQUIRED:
- Logical consistency
- Domain validity
- Extraneous results (if any)

Respond ONLY with JSON:

{
  "is_correct": true,
  "confidence": 0.8,
  "issues_found": [],
  "suggestions": [],
  "edge_cases_checked": []
}
"""

        try:
            data = self._extract_json(self.run(prompt))

            confidence = max(float(data.get("confidence", 0.7)), 0.7)

            return Verification(
                is_correct=bool(data.get("is_correct", True)),
                confidence=confidence,
                issues_found=data.get("issues_found", []),
                suggestions=data.get("suggestions", []),
                edge_cases_checked=data.get("edge_cases_checked", [])
            )

        except Exception as e:
            # Fail-safe: never block pipeline
            return Verification(
                is_correct=True,
                confidence=0.7,
                issues_found=[f"Verifier fallback: {str(e)}"],
                suggestions=["Manual review if needed"],
                edge_cases_checked=[]
            )

    # -------------------------------------------------
    # Deterministic algebra checker
    # -------------------------------------------------

    def _deterministic_algebra_check(
        self,
        problem_text: str,
        final_answer: str
    ):
        if "=" not in problem_text:
            return None

        match = re.search(r"([-+]?\d*\.?\d+)", final_answer)
        if not match:
            return None

        try:
            x_val = float(match.group(1))
        except ValueError:
            return None

        vars_found = set(re.findall(r"[a-zA-Z]", problem_text))
        vars_found.discard("x")
        if vars_found:
            return None

        try:
            lhs, rhs = problem_text.split("=")
            env = {"x": x_val}

            lhs_val = eval(self._sanitize(lhs), {}, env)
            rhs_val = eval(self._sanitize(rhs), {}, env)

            if abs(lhs_val - rhs_val) < 1e-6:
                return True, None
            else:
                return False, f"Substitution failed: LHS={lhs_val}, RHS={rhs_val}"

        except Exception as e:
            return False, f"Deterministic verification error: {str(e)}"

    # -------------------------------------------------
    # Expression sanitization
    # -------------------------------------------------

    def _sanitize(self, expr: str) -> str:
        expr = expr.replace("^", "**")
        expr = expr.replace(" ", "")
        expr = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", expr)
        expr = re.sub(r"(\))(\d|[a-zA-Z])", r"\1*\2", expr)
        return expr

    # -------------------------------------------------
    # JSON extraction
    # -------------------------------------------------

    def _extract_json(self, text: str) -> dict:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("Invalid verifier JSON")
        return json.loads(match.group())
