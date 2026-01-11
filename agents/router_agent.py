"""
Router Agent using Agno framework.
Determines solution strategy and required tools.
"""

from agno.agent import Agent
from agents.base_agent import BaseAgent, get_model
from models.schemas import ParsedProblem
from typing import Any
import json
import re


class RouterAgent(BaseAgent):
    """
    Agent that routes problems to appropriate solution strategies.
    """
    
    def __init__(self):
        super().__init__(
            name="Router",
            description="Routes math problems to optimal solution strategies",
            instructions=[
                "You are a mathematical problem router.",
                "Analyze the problem and determine the best approach.",
                "Select appropriate tools for the solution.",
                "Estimate difficulty level.",
                "Always respond with valid JSON."
            ]
        )
    
    def process(self, input_data: Any) -> dict:
        """Route a parsed problem"""
        if isinstance(input_data, ParsedProblem):
            return self.route(input_data)
        raise ValueError(f"Expected ParsedProblem, got {type(input_data)}")
    
    def route(self, parsed_problem: ParsedProblem) -> dict:
        """Determine solution strategy"""
        
        prompt = f"""Analyze this math problem and determine the solution approach.

PROBLEM: {parsed_problem.problem_text}
TOPIC: {parsed_problem.topic.value}
VARIABLES: {parsed_problem.variables}
CONSTRAINTS: {parsed_problem.constraints}
QUESTION TYPE: {parsed_problem.question_type}

Available tools:
- calculate: Evaluate numerical expressions
- solve_equation: Solve algebraic equations
- differentiate: Find derivatives
- integrate: Find integrals
- simplify_expression: Simplify expressions
- factor_expression: Factor polynomials
- evaluate_limit: Calculate limits
- matrix_operation: Matrix computations
- execute_python: Run Python code

Respond with ONLY a JSON object:
{{
    "confirmed_topic": "algebra|calculus|probability|linear_algebra",
    "strategy": "direct_computation|algebraic_manipulation|formula_application|step_by_step_derivation",
    "tools_needed": ["tool1", "tool2"],
    "estimated_difficulty": "easy|medium|hard",
    "special_considerations": ["list of notes"],
    "recommended_approach": "brief description"
}}"""

        try:
            response = self.run(prompt)
            return self._extract_json(response)
        
        except Exception as e:
            return {
                "confirmed_topic": parsed_problem.topic.value,
                "strategy": "step_by_step_derivation",
                "tools_needed": ["calculate", "solve_equation"],
                "estimated_difficulty": "medium",
                "special_considerations": [f"Routing error: {str(e)}"],
                "recommended_approach": "Apply standard methods"
            }
    
    def _extract_json(self, text: str) -> dict:
        """Extract JSON from response"""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                return json.loads(match.group())
            raise