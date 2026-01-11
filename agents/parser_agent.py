"""
Parser Agent using Agno framework.
Responsible for validating and structuring math input.
"""

from agents.base_agent import BaseAgent
from models.schemas import RawInput, ParsedProblem, MathTopic
from typing import Any
import json
import re


# -------------------------------
# Math semantics (JEE-grade)
# -------------------------------

MATH_CONSTANTS = {"i", "e", "pi"}  # imaginary unit, Euler, π


# -------------------------------
# Validation helpers (NO LLM)
# -------------------------------

def _has_balanced_brackets(text: str) -> bool:
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


def _looks_like_math(text: str) -> bool:
    """
    Detects whether input is a math problem.
    Supports equations, word problems, and probability questions.
    """
    text_lower = text.lower()

    # 1. Symbolic math
    math_symbols = ["=", "+", "-", "*", "/", "^", "√"]
    if any(sym in text for sym in math_symbols):
        return True

    # 2. Numeric presence (counts, quantities)
    if re.search(r"\d", text):
        return True

    # 3. Word-problem / probability cues
    math_keywords = [
        "probability", "chance",
        "how many", "number of", "total",
        "sum", "difference", "product",
        "ratio", "fraction", "percent",
        "average", "mean",
        "container", "bag", "balls", "cards",
        "dice", "coin", "pick", "choose",
        "select", "random"
    ]

    if any(keyword in text_lower for keyword in math_keywords):
        return True

    return False


def _has_invalid_symbols(text: str) -> bool:
    """
    Allows normal English + math.
    Blocks emojis, control chars, code, etc.
    """
    return bool(
        re.search(
            r"[^a-zA-Z0-9\s\+\-\*/=\^\(\)\[\]\{\}\.,√%?:;\'\"<>|\\n\\t]",
            text
        )
    )


# -------------------------------
# Parser Agent
# -------------------------------

class ParserAgent(BaseAgent):
    """
    Strict, domain-aware math parser.
    Produces a ParsedProblem or clarification request.
    """

    def __init__(self):
        super().__init__(
            name="Parser",
            description="Validates and structures math problems",
            instructions=[
                "You are a strict math parser.",
                "ONLY output valid JSON.",
                "Do NOT explain anything.",
                "If input is unclear, mark needs_clarification = true."
            ],
            tools=[]  # no tools → safe for Groq
        )

    # --------------------------------
    # Public Agent Entry Point
    # --------------------------------
    def process(self, input_data: Any) -> ParsedProblem:
        if not isinstance(input_data, RawInput):
            raise TypeError(
                f"ParserAgent expected RawInput, got {type(input_data).__name__}"
            )
        return self._parse(input_data)

    # --------------------------------
    # Core Parsing Logic
    # --------------------------------
    def _parse(self, raw_input: RawInput) -> ParsedProblem:
        text = raw_input.content.strip()

        # -------------------------------
        # HARD VALIDATION (NO LLM)
        # -------------------------------

        if not text:
            return self._clarify(
                text,
                "Input is empty. Please enter a valid math problem."
            )

        if not _looks_like_math(text):
            return self._clarify(
                text,
                "Input does not appear to be a math problem."
            )

        if not _has_balanced_brackets(text):
            return self._clarify(
                text,
                "Unbalanced brackets or parentheses detected."
            )

        if _has_invalid_symbols(text):
            return self._clarify(
                text,
                "Invalid or unclear symbols detected. Please rewrite the problem."
            )

        # -------------------------------
        # LLM STRUCTURING
        # -------------------------------

        prompt = f"""
Convert the following math problem into structured JSON.

RULES:
- Output ONLY JSON
- Do NOT add explanations
- Do NOT include markdown
- Use one of: algebra, calculus, probability, linear_algebra, unknown

INPUT:
{text}

JSON FORMAT:
{{
  "problem_text": "...",
  "topic": "...",
  "variables": ["x"],
  "needs_clarification": false,
  "confidence": 0.9
}}
"""

        response = self.run(prompt)

        if not isinstance(response, str) or not response.strip():
            raise RuntimeError("Parser LLM returned empty or invalid response")

        try:
            data = self._extract_json(response)
        except Exception as e:
            raise RuntimeError(
                f"Parser JSON extraction failed: {str(e)}"
            ) from e

        # -------------------------------
        # Topic Mapping
        # -------------------------------

        topic_map = {
            "algebra": MathTopic.ALGEBRA,
            "calculus": MathTopic.CALCULUS,
            "probability": MathTopic.PROBABILITY,
            "linear_algebra": MathTopic.LINEAR_ALGEBRA,
        }

        topic = topic_map.get(
            str(data.get("topic", "")).lower(),
            MathTopic.UNKNOWN
        )

        # -------------------------------
        # Variable Semantic Correction
        # -------------------------------

        variables = data.get("variables", [])
        content_lower = text.lower()

        cleaned_vars = []
        for v in variables:
            if not isinstance(v, str):
                continue

            v_lower = v.lower()

            if v_lower in MATH_CONSTANTS:
                if any(
                    phrase in content_lower
                    for phrase in (
                        f"let {v_lower}",
                        f"where {v_lower}",
                        f"assume {v_lower}",
                    )
                ):
                    cleaned_vars.append(v)
            else:
                cleaned_vars.append(v)

        return ParsedProblem(
            problem_text=data.get("problem_text", text),
            topic=topic,
            variables=sorted(set(cleaned_vars)),
            needs_clarification=bool(data.get("needs_clarification", False)),
            clarification_reason=None,
            confidence=min(
                max(float(data.get("confidence", 0.8)), 0.0),
                1.0
            ),
        )

    # --------------------------------
    # Clarification Helper
    # --------------------------------
    def _clarify(self, text: str, reason: str) -> ParsedProblem:
        return ParsedProblem(
            problem_text=text,
            topic=MathTopic.UNKNOWN,
            variables=[],
            needs_clarification=True,
            clarification_reason=reason,
            confidence=0.3
        )

    # --------------------------------
    # JSON Extraction Utility
    # --------------------------------
    def _extract_json(self, text: str) -> dict:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON object found in LLM output")
        return json.loads(match.group())
