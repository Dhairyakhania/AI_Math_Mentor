"""
JEE-grade Solver Agent using Agno.
Handles direct math and word problems with strict structure.
RAG-COMPLIANT + TYPE-SAFE VERSION.
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
    # ENTRY POINT
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

        # -------------------------------------------------
        # HARD BLOCK: NO RAG CONTEXT → NO SOLVE
        # -------------------------------------------------
        if not context:
            return Solution(
                final_answer="Insufficient reference material to solve reliably.",
                steps=[],
                method_used="rag_blocked",
                context_used=[],
                tool_calls=[],
                confidence=0.2
            )

        problem_type = routing_info.get("problem_type", "direct_math")
        equations = routing_info.get("equations", [])
        variables = routing_info.get("variables", {})
        constraints = routing_info.get("constraints", [])
        memory_hints = routing_info.get("memory_hints", [])

        # -------------------------------------------------
        # PROMPT CONSTRUCTION
        # -------------------------------------------------

        prompt = f"""
Solve the following JEE-level mathematics problem.

PROBLEM:
{parsed_problem.problem_text}
"""

        # -------------------------------------------------
        # INJECT RAG CONTEXT (MANDATORY)
        # -------------------------------------------------

        prompt += "\nREFERENCE MATERIAL (USE ONLY IF RELEVANT):\n"

        for i, ctx in enumerate(context, start=1):
            # Defensive: context may not yet be normalized
            source = getattr(ctx, "source", "unknown")
            text = getattr(ctx, "text", "")

            prompt += f"""
REFERENCE {i}:
SOURCE: {source}
CONTENT:
{text}
"""

        # -------------------------------------------------
        # MEMORY-AWARE AUGMENTATION
        # -------------------------------------------------

        if memory_hints:
            prompt += "\nKNOWN PITFALLS FROM PAST ATTEMPTS:\n"
            for hint in memory_hints:
                prompt += f"- {hint}\n"

        # -------------------------------------------------
        # PROBLEM TYPE HANDLING
        # -------------------------------------------------

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

        # -------------------------------------------------
        # STRICT SOLVING RULES
        # -------------------------------------------------

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
- Use reference material only when relevant

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

        # -------------------------------------------------
        # STEP EXTRACTION
        # -------------------------------------------------

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

        if not steps:
            steps.append(
                SolutionStep(
                    step_number=1,
                    description="Solution derived using standard JEE methods."
                )
            )

        # -------------------------------------------------
        # FINAL ANSWER EXTRACTION
        # -------------------------------------------------

        final_answer = self._extract_final_answer(response)

        # -------------------------------------------------
        # CONTEXT NORMALIZATION (CRITICAL FIX)
        # -------------------------------------------------

        normalized_context: List[RetrievedContext] = []

        for item in context:
            # Already correct type
            if isinstance(item, RetrievedContext):
                normalized_context.append(item)

            # Dict leaked in → normalize safely
            elif isinstance(item, dict):
                try:
                    normalized_context.append(
                        RetrievedContext(
                            text=item.get("text", ""),
                            source=item.get("source", "unknown"),
                            relevance_score=float(item.get("relevance_score", 0.0)),
                            metadata=item.get("metadata", {})
                        )
                    )
                except Exception:
                    # Skip malformed context entries safely
                    continue

        # -------------------------------------------------
        # CONFIDENCE CALIBRATION
        # -------------------------------------------------

        if normalized_context:
            avg_relevance = sum(
                c.relevance_score for c in normalized_context
            ) / len(normalized_context)
        else:
            avg_relevance = 0.3

        confidence = min(0.95, max(0.3, avg_relevance))

        # -------------------------------------------------
        # FINAL SOLUTION
        # -------------------------------------------------

        return Solution(
            final_answer=final_answer,
            steps=steps,
            method_used="jee_structured_rag_reasoning",
            context_used=normalized_context,
            tool_calls=[],
            confidence=confidence
        )

    # ------------------------------------------------------------------
    # UTILITIES
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
