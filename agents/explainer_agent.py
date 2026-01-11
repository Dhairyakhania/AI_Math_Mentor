"""
Explainer Agent using Agno framework.
Consumes ACTION/RESULT steps from SolverAgent.
"""

from agents.base_agent import BaseAgent
from models.schemas import ParsedProblem, Solution, Verification, Explanation
from typing import Any
import json
import re


class ExplainerAgent(BaseAgent):
    """Explains solver steps clearly without recomputation."""

    def __init__(self):
        super().__init__(
            name="Explainer",
            description="Explains solver steps in a student-friendly way",
            instructions=[
                "You are a JEE-level math tutor.",
                "Explain WHY each solver step is valid.",
                "Do NOT redo calculations.",
                "Do NOT add new steps.",
                "Do NOT use formulas unless already implied.",
                "Do NOT output markdown or LaTeX.",
                "Output ONLY valid JSON."
            ],
            tools=[]  # 🚫 absolutely no tools
        )

    def process(self, input_data: Any) -> Explanation:
        if not (isinstance(input_data, tuple) and len(input_data) == 3):
            raise ValueError("Expected (ParsedProblem, Solution, Verification)")
        return self.explain(*input_data)

    def explain(
        self,
        parsed_problem: ParsedProblem,
        solution: Solution,
        verification: Verification
    ) -> Explanation:

        # ---------- Step extraction (STRICT & SAFE) ----------
        structured_steps = []
        for step in solution.steps:
            action = ""
            result = ""

            action_match = re.search(r"ACTION:\s*(.+)", step.description, re.IGNORECASE)
            result_match = re.search(r"RESULT:\s*(.+)", step.description, re.IGNORECASE)

            if action_match:
                action = action_match.group(1).strip()
            if result_match:
                result = result_match.group(1).strip()

            structured_steps.append({
                "step_number": step.step_number,
                "action": action,
                "result": result
            })

        if not structured_steps:
            return self._fallback(parsed_problem, solution)

        steps_str = "\n".join(
            f"Step {s['step_number']}:\n"
            f"Action performed: {s['action']}\n"
            f"Result obtained: {s['result']}"
            for s in structured_steps
        )

        # ---------- Prompt ----------
        prompt = f"""
Explain the following solved math problem to a student.

PROBLEM:
{parsed_problem.problem_text}

SOLVER STEPS (already computed):
{steps_str}

FINAL ANSWER:
{solution.final_answer}

VERIFICATION STATUS:
{"Correct" if verification.is_correct else "Uncertain"}

RULES:
- Explain the idea behind each step
- Match steps exactly
- No calculations
- No formulas unless obvious
- Simple, clear language

Return ONLY JSON:

{{
  "summary": "Brief overview of how the problem was solved",
  "detailed_steps": [
    "Explanation of Step 1",
    "Explanation of Step 2"
  ],
  "key_concepts": ["concept1", "concept2"],
  "common_mistakes_to_avoid": ["mistake1", "mistake2"],
  "related_problems": ["problem type"],
  "encouragement": "Short motivational message"
}}
"""

        try:
            response = self.run(prompt)
            data = self._extract_json(response)

            return Explanation(
                summary=data.get("summary", ""),
                detailed_steps=data.get("detailed_steps", []),
                key_concepts=data.get("key_concepts", []),
                common_mistakes_to_avoid=data.get("common_mistakes_to_avoid", []),
                related_problems=data.get("related_problems", []),
                encouragement=data.get("encouragement", "")
            )

        except Exception:
            return self._fallback(parsed_problem, solution)

    # ---------- Fallback (never breaks UI) ----------
    def _fallback(self, parsed_problem: ParsedProblem, solution: Solution) -> Explanation:
        return Explanation(
            summary=f"The problem was solved step by step to find {solution.final_answer}.",
            detailed_steps=[
                f"Step {s.step_number}: {s.description}"
                for s in solution.steps
            ],
            key_concepts=[parsed_problem.topic.value],
            common_mistakes_to_avoid=[],
            related_problems=[],
            encouragement="Keep practicing — structured thinking improves accuracy."
        )

    # ---------- JSON extraction ----------
    def _extract_json(self, text: str) -> dict:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON object found")
        return json.loads(match.group())
