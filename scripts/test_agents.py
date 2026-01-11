"""
Test script for validating agent system functionality.
Run with: python -m scripts.test_agents
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from unittest.mock import Mock, patch
import json

from models.schemas import (
    RawInput, InputType, ParsedProblem, MathTopic,
    Solution, Verification, Explanation, FinalResult
)
from config import Config


class TestParserAgent(unittest.TestCase):
    """Test cases for Parser Agent"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        from agents.parser_agent import ParserAgent
        cls.parser = ParserAgent()
    
    def test_parse_simple_equation(self):
        """Test parsing a simple quadratic equation"""
        raw_input = RawInput(
            type=InputType.TEXT,
            content="Solve x^2 - 5x + 6 = 0",
            confidence=1.0
        )
        
        result = self.parser.parse(raw_input)
        
        self.assertIsInstance(result, ParsedProblem)
        self.assertIn("x", result.problem_text.lower())
        self.assertEqual(result.topic, MathTopic.ALGEBRA)
        self.assertIn("x", result.variables)
        self.assertFalse(result.needs_clarification)
    
    def test_parse_calculus_problem(self):
        """Test parsing a calculus problem"""
        raw_input = RawInput(
            type=InputType.TEXT,
            content="Find the derivative of f(x) = x^3 + 2x^2 - x + 5",
            confidence=1.0
        )
        
        result = self.parser.parse(raw_input)
        
        self.assertEqual(result.topic, MathTopic.CALCULUS)
        self.assertIn("derivative", result.problem_text.lower())
    
    def test_parse_probability_problem(self):
        """Test parsing a probability problem"""
        raw_input = RawInput(
            type=InputType.TEXT,
            content="What is the probability of getting exactly 3 heads in 5 coin tosses?",
            confidence=1.0
        )
        
        result = self.parser.parse(raw_input)
        
        self.assertEqual(result.topic, MathTopic.PROBABILITY)
    
    def test_parse_ambiguous_input(self):
        """Test that ambiguous input triggers clarification"""
        raw_input = RawInput(
            type=InputType.TEXT,
            content="solve for something",
            confidence=0.5
        )
        
        result = self.parser.parse(raw_input)
        
        # Should flag as needing clarification due to vague input
        self.assertTrue(result.needs_clarification or result.confidence < 0.7)
    
    def test_parse_with_constraints(self):
        """Test parsing problem with constraints"""
        raw_input = RawInput(
            type=InputType.TEXT,
            content="Find x if x^2 = 16, where x > 0",
            confidence=1.0
        )
        
        result = self.parser.parse(raw_input)
        
        self.assertIn("x", result.variables)
        # Should identify the constraint x > 0
        self.assertTrue(
            any(">" in c or "positive" in c.lower() for c in result.constraints) or
            "x > 0" in result.problem_text
        )


class TestRouterAgent(unittest.TestCase):
    """Test cases for Router Agent"""
    
    @classmethod
    def setUpClass(cls):
        from agents.router_agent import RouterAgent
        cls.router = RouterAgent()
    
    def test_route_algebra_problem(self):
        """Test routing an algebra problem"""
        parsed = ParsedProblem(
            problem_text="Solve x^2 - 5x + 6 = 0",
            topic=MathTopic.ALGEBRA,
            variables=["x"],
            question_type="solve"
        )
        
        result = self.router.route(parsed)
        
        self.assertIn("confirmed_topic", result)
        self.assertIn("strategy", result)
        self.assertIn("tools_needed", result)
    
    def test_route_calculus_problem(self):
        """Test routing a calculus problem"""
        parsed = ParsedProblem(
            problem_text="Find the derivative of x^3",
            topic=MathTopic.CALCULUS,
            variables=["x"],
            question_type="differentiate"
        )
        
        result = self.router.route(parsed)
        
        self.assertEqual(result.get("confirmed_topic"), "calculus")
        self.assertIn("differentiate", result.get("tools_needed", []))


class TestSolverAgent(unittest.TestCase):
    """Test cases for Solver Agent"""
    
    @classmethod
    def setUpClass(cls):
        from agents.solver_agent import SolverAgent
        cls.solver = SolverAgent()
    
    def test_solve_quadratic(self):
        """Test solving a quadratic equation"""
        parsed = ParsedProblem(
            problem_text="Solve x^2 - 5x + 6 = 0",
            topic=MathTopic.ALGEBRA,
            variables=["x"],
            question_type="solve"
        )
        
        routing_info = {
            "confirmed_topic": "algebra",
            "strategy": "algebraic_manipulation",
            "tools_needed": ["solve_equation"]
        }
        
        context = []
        
        result = self.solver.solve(parsed, routing_info, context)
        
        self.assertIsInstance(result, Solution)
        self.assertIsNotNone(result.final_answer)
        # Should find x = 2 and x = 3
        self.assertTrue(
            "2" in result.final_answer and "3" in result.final_answer
        )
    
    def test_solve_derivative(self):
        """Test finding a derivative"""
        parsed = ParsedProblem(
            problem_text="Find the derivative of f(x) = x^3",
            topic=MathTopic.CALCULUS,
            variables=["x"],
            question_type="differentiate"
        )
        
        routing_info = {
            "confirmed_topic": "calculus",
            "strategy": "formula_application",
            "tools_needed": ["differentiate"]
        }
        
        result = self.solver.solve(parsed, routing_info, [])
        
        self.assertIsNotNone(result.final_answer)
        # Derivative of x^3 is 3x^2
        self.assertTrue("3" in result.final_answer)


class TestVerifierAgent(unittest.TestCase):
    """Test cases for Verifier Agent"""
    
    @classmethod
    def setUpClass(cls):
        from agents.verifier_agent import VerifierAgent
        cls.verifier = VerifierAgent()
    
    def test_verify_correct_solution(self):
        """Test verifying a correct solution"""
        parsed = ParsedProblem(
            problem_text="Solve x^2 - 5x + 6 = 0",
            topic=MathTopic.ALGEBRA,
            variables=["x"]
        )
        
        solution = Solution(
            final_answer="x = 2 or x = 3",
            steps=[],
            method_used="factoring"
        )
        
        result = self.verifier.verify(parsed, solution)
        
        self.assertIsInstance(result, Verification)
        self.assertTrue(result.is_correct)
        self.assertGreater(result.confidence, 0.7)
    
    def test_verify_incorrect_solution(self):
        """Test verifying an incorrect solution"""
        parsed = ParsedProblem(
            problem_text="Solve x^2 - 5x + 6 = 0",
            topic=MathTopic.ALGEBRA,
            variables=["x"]
        )
        
        solution = Solution(
            final_answer="x = 1",  # Wrong answer
            steps=[],
            method_used="guessing"
        )
        
        result = self.verifier.verify(parsed, solution)
        
        # Should detect the error or have low confidence
        self.assertTrue(
            not result.is_correct or 
            result.confidence < 0.8 or 
            len(result.issues_found) > 0
        )


class TestExplainerAgent(unittest.TestCase):
    """Test cases for Explainer Agent"""
    
    @classmethod
    def setUpClass(cls):
        from agents.explainer_agent import ExplainerAgent
        cls.explainer = ExplainerAgent()
    
    def test_generate_explanation(self):
        """Test generating an explanation"""
        parsed = ParsedProblem(
            problem_text="Solve x^2 - 5x + 6 = 0",
            topic=MathTopic.ALGEBRA,
            variables=["x"]
        )
        
        solution = Solution(
            final_answer="x = 2 or x = 3",
            steps=[],
            method_used="factoring"
        )
        
        verification = Verification(
            is_correct=True,
            confidence=0.95
        )
        
        result = self.explainer.explain(parsed, solution, verification)
        
        self.assertIsInstance(result, Explanation)
        self.assertIsNotNone(result.summary)
        self.assertGreater(len(result.summary), 0)


class TestLeaderAgent(unittest.TestCase):
    """Test cases for Leader Agent (full workflow)"""
    
    @classmethod
    def setUpClass(cls):
        from agents.leader_agent import LeaderAgent
        cls.leader = LeaderAgent()
    
    def test_full_workflow_simple_problem(self):
        """Test the complete workflow with a simple problem"""
        raw_input = RawInput(
            type=InputType.TEXT,
            content="What is 2 + 2?",
            confidence=1.0
        )
        
        result = self.leader.solve(raw_input)
        
        self.assertIsInstance(result, FinalResult)
        self.assertIn(result.status, ["success", "needs_hitl"])
        
        if result.status == "success":
            self.assertIsNotNone(result.solution)
            self.assertIn("4", result.solution.final_answer)
    
    def test_full_workflow_quadratic(self):
        """Test the complete workflow with a quadratic equation"""
        raw_input = RawInput(
            type=InputType.TEXT,
            content="Solve: x^2 - 4 = 0",
            confidence=1.0
        )
        
        result = self.leader.solve(raw_input)
        
        self.assertIsInstance(result, FinalResult)
        
        if result.status == "success":
            # Should find x = 2 and x = -2
            answer = result.solution.final_answer.lower()
            self.assertTrue("2" in answer)


class TestTools(unittest.TestCase):
    """Test cases for calculation tools"""
    
    def test_calculate(self):
        """Test basic calculation"""
        from tools.calculator import calculate
        
        result = calculate("2 + 3 * 4")
        self.assertIn("14", result)
    
    def test_solve_equation(self):
        """Test equation solving"""
        from tools.calculator import solve_equation
        
        result = solve_equation("x**2 - 4", "x")
        self.assertIn("2", result)
        self.assertIn("-2", result)
    
    def test_differentiate(self):
        """Test differentiation"""
        from tools.calculator import differentiate
        
        result = differentiate("x**3", "x")
        self.assertIn("3", result)
        self.assertIn("x", result)
    
    def test_integrate(self):
        """Test integration"""
        from tools.calculator import integrate
        
        result = integrate("2*x", "x")
        self.assertIn("x**2", result.replace(" ", "").replace("^", "**"))
    
    def test_simplify(self):
        """Test expression simplification"""
        from tools.calculator import simplify_expression
        
        result = simplify_expression("(x+1)**2 - x**2 - 2*x - 1")
        self.assertIn("0", result)
    
    def test_factor(self):
        """Test factoring"""
        from tools.calculator import factor_expression
        
        result = factor_expression("x**2 - 4")
        self.assertIn("x - 2", result.replace(" ", ""))
        self.assertIn("x + 2", result.replace(" ", ""))
    
    def test_limit(self):
        """Test limit evaluation"""
        from tools.calculator import evaluate_limit
        
        result = evaluate_limit("sin(x)/x", "x", "0")
        self.assertIn("1", result)


class TestRAG(unittest.TestCase):
    """Test cases for RAG pipeline"""
    
    @classmethod
    def setUpClass(cls):
        from rag.vectorstore import MathKnowledgeBase
        cls.kb = MathKnowledgeBase()
    
    def test_search_knowledge_base(self):
        """Test knowledge base search"""
        results = self.kb.search("quadratic formula", k=3)
        
        self.assertIsInstance(results, list)
        # Should find relevant content if KB is loaded
    
    def test_search_by_topic(self):
        """Test topic-filtered search"""
        results = self.kb.search_by_topic("derivative rules", "calculus", k=3)
        
        self.assertIsInstance(results, list)


class TestMemory(unittest.TestCase):
    """Test cases for memory system"""
    
    @classmethod
    def setUpClass(cls):
        # Use a test database
        cls.test_db_path = "./data/test_memory.db"
        from memory.store import MemoryStore
        cls.memory = MemoryStore(cls.test_db_path)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test database"""
        import os
        if os.path.exists(cls.test_db_path):
            os.remove(cls.test_db_path)
    
    def test_save_interaction(self):
        """Test saving an interaction"""
        from models.schemas import MemoryEntry
        
        entry = MemoryEntry(
            input_type=InputType.TEXT,
            raw_input="Test problem",
            parsed_problem="Parsed test",
            topic=MathTopic.ALGEBRA,
            solution="Test solution",
            verification_score=0.9
        )
        
        interaction_id = self.memory.save_interaction(entry)
        
        self.assertIsInstance(interaction_id, int)
        self.assertGreater(interaction_id, 0)
    
    def test_find_similar_problems(self):
        """Test finding similar problems"""
        similar = self.memory.find_similar_problems("quadratic equation", k=3)
        
        self.assertIsInstance(similar, list)
    
    def test_feedback_stats(self):
        """Test getting feedback statistics"""
        stats = self.memory.get_feedback_stats()
        
        self.assertIsInstance(stats, dict)


class TestProcessors(unittest.TestCase):
    """Test cases for input processors"""
    
    def test_text_processor(self):
        """Test text processor"""
        from processors.text_processor import TextProcessor
        
        processor = TextProcessor()
        result = processor.process("Solve x^2 = 4")
        
        self.assertEqual(result.type, InputType.TEXT)
        self.assertIn("x", result.content)
    
    def test_text_processor_latex(self):
        """Test text processor with LaTeX"""
        from processors.text_processor import TextProcessor
        
        processor = TextProcessor()
        result = processor.process(r"Solve $\frac{x}{2} = 3$")
        
        self.assertEqual(result.type, InputType.TEXT)
        # LaTeX should be converted


class TestIntegration(unittest.TestCase):
    """Integration tests for the full system"""
    
    @classmethod
    def setUpClass(cls):
        from agents.team import MathMentorTeam
        cls.team = MathMentorTeam()
    
    def test_end_to_end_simple(self):
        """Test end-to-end with simple problem"""
        raw_input = RawInput(
            type=InputType.TEXT,
            content="Calculate 15% of 200",
            confidence=1.0
        )
        
        result = self.team.solve(raw_input)
        
        self.assertIsNotNone(result)
        self.assertIn(result.status, ["success", "needs_hitl", "error"])
    
    def test_end_to_end_algebra(self):
        """Test end-to-end with algebra problem"""
        raw_input = RawInput(
            type=InputType.TEXT,
            content="If 2x + 5 = 13, find x",
            confidence=1.0
        )
        
        result = self.team.solve(raw_input)
        
        self.assertIsNotNone(result)
        if result.status == "success":
            # x should be 4
            self.assertIn("4", result.solution.final_answer)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestParserAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestRouterAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestSolverAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestVerifierAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestExplainerAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestLeaderAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestTools))
    suite.addTests(loader.loadTestsFromTestCase(TestRAG))
    suite.addTests(loader.loadTestsFromTestCase(TestMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestProcessors))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


def run_quick_test():
    """Run a quick smoke test"""
    print("Running quick smoke test...")
    
    try:
        # Test imports
        print("  Testing imports...")
        from agents.team import MathMentorTeam
        from models.schemas import RawInput, InputType
        from processors.text_processor import TextProcessor
        from tools.calculator import calculate, solve_equation
        print("  ✓ Imports successful")
        
        # Test calculator tools
        print("  Testing calculator tools...")
        result = calculate("2 + 2")
        assert "4" in result, f"Expected 4, got {result}"
        print("  ✓ Calculator working")
        
        # Test equation solver
        print("  Testing equation solver...")
        result = solve_equation("x**2 - 4", "x")
        assert "2" in result, f"Expected 2 in result, got {result}"
        print("  ✓ Equation solver working")
        
        # Test text processor
        print("  Testing text processor...")
        processor = TextProcessor()
        result = processor.process("Solve x^2 = 4")
        assert result.type == InputType.TEXT
        print("  ✓ Text processor working")
        
        # Test full pipeline (if API key available)
        if Config.OPENAI_API_KEY:
            print("  Testing full pipeline...")
            team = MathMentorTeam()
            raw_input = RawInput(
                type=InputType.TEXT,
                content="What is 5 + 7?",
                confidence=1.0
            )
            result = team.solve(raw_input)
            assert result is not None
            print(f"  ✓ Full pipeline working (status: {result.status})")
        else:
            print("  ⚠ Skipping full pipeline test (no API key)")
        
        print("\n✅ All smoke tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Math Mentor agents")
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run quick smoke test only"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        sys.exit(run_quick_test())
    else:
        sys.exit(run_tests())