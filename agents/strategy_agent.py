"""
Strategy Agent for JEE-level problem planning.
Uses Agno framework.
"""

from agents.base_agent import BaseAgent
from models.schemas import ParsedProblem
from typing import Any, Dict
import json
import re


class StrategyAgent(BaseAgent):
    """
    Plans HOW to solve a problem.
    Performs NO computation.
    """

    def __init__(self):
        super().__init__(
            name="Strategy",
            description="Plans solution strategy for complex math problems",
            instructions=[
                "You are an expert JEE problem-solving strategist.",
                "Your task is to decide HOW to solve the problem, not to solve it.",
                "Do NOT compute, simplify, or evaluate expressions.",
                "Do NOT call tools.",
                "Identify cases, constraints, and logical order of steps.",
                "Always respond with valid JSON only."
            ]
        )

    def process(self, input_data: Any) -> Dict:
        if not isinstance(input_data, ParsedProblem):
            raise ValueError("StrategyAgent expects ParsedProblem")

        prompt = f"""
Analyze the following math problem and design a solution strategy.

PROBLEM:
{input_data.problem_text}

TOPIC:
{input_data.topic.value}

RULES:
- Do NOT solve the problem
- Do NOT compute anything
- Do NOT simplify expressions

Respond ONLY with JSON:

{{
  "core_concepts": [],
  "problem_type": "direct | multi_case | parameter_based | inequality | geometry_based | mixed",
  "requires_case_split": true or false,
  "cases": [],
  "strategy_steps": [],
  "what_solver_should_compute": [],
  "common_traps": [],
  "jee_insights": []
}}
"""
        response = self.run(prompt)
        return self._extract_json(response)

    def _extract_json(self, text: str) -> Dict:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {
                "core_concepts": [],
                "problem_type": "unknown",
                "requires_case_split": False,
                "cases": [],
                "strategy_steps": [],
                "what_solver_should_compute": [],
                "common_traps": [],
                "jee_insights": []
            }

        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {
                "core_concepts": [],
                "problem_type": "unknown",
                "requires_case_split": False,
                "cases": [],
                "strategy_steps": [],
                "what_solver_should_compute": [],
                "common_traps": [],
                "jee_insights": []
            }
