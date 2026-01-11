"""
JEE-grade Solver Agent using Agno.
Handles direct math and word problems with strict structure.
"""

from agents.base_agent import BaseAgent
from models.schemas import ParsedProblem, Solution, SolutionStep, RetrievedContext
from tools.calculator import calculate, solve_equation
from typing import Any, List
import re


class SolverAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Solver",
            description="Solves JEE-style math problems with structured reasoning",
            instructions=[
                "You are a JEE-level mathematics solver.",
                "Never guess equations.",
                "Never skip steps.",
                "For word problems: form equations first, then solve.",
                "Explicitly state the mathematical principle used in each step.",
                "Avoid conversational language.",
                "Output strictly in the required STEP format.",
                "Do NOT output markdown or LaTeX."
            ],
            tools=[calculate, solve_equation]
        )

    # ------------------------------------------------------------------
    # ENTRY POINT (agent-compatible)
    # ------------------------------------------------------------------

    def process(self, input_data: Any) -> Solution:
        """
        Expected input:
        (ParsedProblem, routing_info: dict, context: list)
        """
        if not isinstance(input_data, (list, tuple)) or len(input_data) != 3:
            raise TypeError(
                "SolverAgent.process expects (ParsedProblem, routing_info, context)"
            )

        parsed_problem, routing_info, context = input_data
        return self.solve(parsed_problem, routing_info, context)

    # ------------------------------------------------------------------
    # CORE SOLVER
    # ------------------------------------------------------------------

    def solve(
        self,
        parsed_problem: ParsedProblem,
        routing_info: dict,
        context: list
    ) -> Solution:

        problem_type = routing_info.get("problem_type", "direct_math")
        equations = routing_info.get("equations", [])
        variables = routing_info.get("variables", {})
        constraints = routing_info.get("constraints", [])

        # ---------------------------
        # Prompt construction
        # ---------------------------

        prompt = f"""
Solve the following JEE-level mathematics problem.

PROBLEM:
{parsed_problem.problem_text}
"""

        if problem_type == "word_problem":
            prompt += f"""

THIS IS A WORD PROBLEM.

STEP 1 MUST:
- Define variables clearly
- Justify equation formation

DEFINED VARIABLES:
{variables}

FORMED EQUATIONS:
{equations}

CONSTRAINTS:
{constraints}
"""
        else:
            prompt += """

THIS IS A DIRECT MATHEMATICAL PROBLEM.
"""

        prompt += """
RULES:
- Each step MUST include:
  PRINCIPLE
  ACTION
  RESULT
- Use standard JEE principles (Vieta, substitution, domain restriction)
- Do NOT explain in prose
- Do NOT skip steps
- Do NOT verify correctness

FORMAT STRICTLY:

STEP 1:
PRINCIPLE: <principle used>
ACTION: <what is done>
RESULT: <result>

STEP 2:
...

FINAL ANSWER:
<answer only>

Begin solving now.
"""

        response = self.run(prompt)

        if not isinstance(response, str) or not response.strip():
            raise RuntimeError("Solver LLM returned empty or invalid response")

        # ---------------------------
        # STEP extraction
        # ---------------------------

        steps: List[SolutionStep] = []

        step_blocks = re.findall(
            r"STEP\s*(\d+):([\s\S]*?)(?=STEP|\Z)",
            response,
            re.IGNORECASE
        )

        for num, block in step_blocks:
            steps.append(
                SolutionStep(
                    step_number=int(num),
                    description=block.strip()
                )
            )

        # Fallback: ensure at least one step exists
        if not steps:
            steps.append(
                SolutionStep(
                    step_number=1,
                    description="Solution derived using standard JEE methods."
                )
            )

        # ---------------------------
        # FINAL ANSWER extraction
        # ---------------------------

        final_answer = self._extract_final_answer(response)

        # ---------------------------
        # Context normalization (CRITICAL)
        # ---------------------------

        normalized_context: List[RetrievedContext] = []

        if context:
            for item in context:
                # Proper RetrievedContext
                if isinstance(item, RetrievedContext):
                    normalized_context.append(item)

                # Strategy dict or other metadata → ignore safely
                elif isinstance(item, dict):
                    continue

        # ---------------------------
        # Final Solution object
        # ---------------------------

        return Solution(
            final_answer=final_answer,
            steps=steps,
            method_used="jee_structured_reasoning",
            context_used=normalized_context,
            tool_calls=[],
            confidence=0.9
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _extract_final_answer(self, text: str) -> str:
        match = re.search(
            r"FINAL ANSWER:\s*(.+)",
            text,
            re.IGNORECASE | re.DOTALL
        )
        if match:
            answer = match.group(1).strip()
            return answer if answer else "Unable to determine"

        return "Unable to determine"
