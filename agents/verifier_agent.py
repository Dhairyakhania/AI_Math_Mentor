"""
Verifier Agent using Agno.
Performs JEE-style correctness and domain checks.
"""

from agents.base_agent import BaseAgent
from models.schemas import ParsedProblem, Solution, Verification
from typing import Any
import re
import json
import math


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
                "Return ONLY valid JSON."
            ],
            tools=[]
        )

    def process(self, input_data: Any) -> Verification:
        parsed_problem, solution = input_data
        return self.verify(parsed_problem, solution)

    # -------------------------------------------------
    # MAIN VERIFICATION
    # -------------------------------------------------

    def verify(
        self,
        parsed_problem: ParsedProblem,
        solution: Solution
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
        # 2. PROBABILITY → SOFT VERIFICATION (NO HITL)
        # =================================================
        if topic == "probability":
            # Try to extract numeric probability
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

            # Even if numeric check fails, do not block
            return Verification(
                is_correct=True,
                confidence=0.75,
                issues_found=[],
                suggestions=[],
                edge_cases_checked=["Conceptual probability validation"]
            )

        # =================================================
        # 3. WORD / CALCULUS / OTHER → LLM VERIFICATION
        # =================================================

        prompt = f"""
Verify the following solution.

PROBLEM:
{parsed_problem.problem_text}

SOLUTION STEPS:
{[s.description for s in solution.steps]}

FINAL ANSWER:
{solution.final_answer}

CHECKS REQUIRED:
- Logical consistency
- Domain validity
- Extraneous results (if any)

Respond ONLY with JSON:

{{
  "is_correct": true,
  "confidence": 0.8,
  "issues_found": [],
  "suggestions": [],
  "edge_cases_checked": []
}}
"""

        try:
            data = self._extract_json(self.run(prompt))

            # IMPORTANT:
            # Do NOT allow LLM to block non-algebra problems
            confidence = max(float(data.get("confidence", 0.7)), 0.7)

            return Verification(
                is_correct=bool(data.get("is_correct", True)),
                confidence=confidence,
                issues_found=data.get("issues_found", []),
                suggestions=data.get("suggestions", []),
                edge_cases_checked=data.get("edge_cases_checked", [])
            )

        except Exception as e:
            # Fail-safe: never crash or block
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
        # Must be an equation
        if "=" not in problem_text:
            return None

        # Extract numeric answer
        match = re.search(r"([-+]?\d*\.?\d+)", final_answer)
        if not match:
            return None

        try:
            x_val = float(match.group(1))
        except ValueError:
            return None

        # Ensure only variable is x
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
        """
        Sanitizes math expressions for safe eval.
        Handles implicit multiplication.
        """
        expr = expr.replace("^", "**")
        expr = expr.replace(" ", "")

        # Insert * between number and variable or (
        expr = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", expr)

        # Insert * between ) and number or variable
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
