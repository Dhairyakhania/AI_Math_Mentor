"""
Leader Agent using Agno framework.
Orchestrates the multi-agent workflow.
"""

from agents.parser_agent import ParserAgent
from agents.strategy_agent import StrategyAgent
from agents.router_agent import RouterAgent
from agents.solver_agent import SolverAgent
from agents.verifier_agent import VerifierAgent
from agents.explainer_agent import ExplainerAgent

from rag.retriever import RAGRetriever

from models.schemas import (
    RawInput, ParsedProblem, Solution, Verification,
    Explanation, FinalResult, AgentTrace
)

from config import Config
from datetime import datetime


class LeaderAgent:
    """
    Orchestrates the Math Mentor workflow.
    Pipeline:
    Parser → Strategy → Router → Retriever → Solver → Verifier → Explainer
    """

    def __init__(self):
        self.parser = ParserAgent()
        self.strategy = StrategyAgent()
        self.router = RouterAgent()
        self.retriever = RAGRetriever()
        self.solver = SolverAgent()
        self.verifier = VerifierAgent()
        self.explainer = ExplainerAgent()

    def solve(self, raw_input: RawInput) -> FinalResult:
        traces: list[AgentTrace] = []

        try:
            # -----------------------------
            # STEP 1: PARSER
            # -----------------------------
            trace = self._create_trace("Parser", raw_input.content[:60])
            traces.append(trace)

            parsed_problem = self.parser.process(raw_input)
            self._complete_trace(trace, f"Topic: {parsed_problem.topic.value}")

            if parsed_problem.needs_clarification:
                return FinalResult(
                    status="needs_hitl",
                    raw_input=raw_input,
                    parsed_problem=parsed_problem,
                    agent_traces=traces,
                    hitl_reason=parsed_problem.clarification_reason
                )

            # -----------------------------
            # STEP 2: STRATEGY
            # -----------------------------
            trace = self._create_trace("Strategy", "Planning solution approach")
            traces.append(trace)

            strategy = self.strategy.process(parsed_problem)
            self._complete_trace(
                trace,
                f"Type: {strategy.get('problem_type', 'unknown')}"
            )

            # -----------------------------
            # STEP 3: ROUTER
            # -----------------------------
            trace = self._create_trace("Router", "Selecting solve strategy")
            traces.append(trace)

            routing_info = self.router.route(parsed_problem)
            self._complete_trace(
                trace,
                routing_info.get("strategy", "default")
            )

            # -----------------------------
            # STEP 4: RETRIEVER (RAG)
            # -----------------------------
            trace = self._create_trace("Retriever", "Fetching reference material")
            traces.append(trace)

            retrieved_context = self.retriever.retrieve(parsed_problem)
            self._complete_trace(
                trace,
                f"Retrieved {len(retrieved_context)} chunks"
            )

            # -----------------------------
            # STEP 5: SOLVER
            # -----------------------------
            trace = self._create_trace("Solver", "Executing solution plan")
            traces.append(trace)

            solution = self.solver.process(
                (parsed_problem, routing_info, retrieved_context)
            )

            self._complete_trace(
                trace,
                solution.final_answer[:60]
                if solution and solution.final_answer
                else "No answer produced"
            )

            # -----------------------------
            # STEP 6: VERIFIER (FIXED)
            # -----------------------------
            trace = self._create_trace("Verifier", "Checking correctness")
            traces.append(trace)

            verification = self.verifier.process(
                (parsed_problem, solution, retrieved_context)
            )

            self._complete_trace(
                trace,
                f"Correct: {verification.is_correct}, "
                f"Conf: {verification.confidence:.0%}"
            )

            # -----------------------------
            # HITL POLICY
            # -----------------------------
            if (
                parsed_problem.topic.value in ("algebra", "linear_algebra")
                and verification.confidence < Config.VERIFIER_CONFIDENCE_THRESHOLD
            ):
                return FinalResult(
                    status="needs_hitl",
                    raw_input=raw_input,
                    parsed_problem=parsed_problem,
                    solution=solution,
                    verification=verification,
                    agent_traces=traces,
                    hitl_reason="Low verifier confidence"
                )

            # -----------------------------
            # STEP 7: EXPLAINER
            # -----------------------------
            trace = self._create_trace("Explainer", "Generating explanation")
            traces.append(trace)

            explanation = self.explainer.explain(
                parsed_problem, solution, verification
            )

            self._complete_trace(
                trace,
                explanation.summary[:60]
                if explanation and explanation.summary
                else "Explanation generated"
            )

            return FinalResult(
                status="success",
                raw_input=raw_input,
                parsed_problem=parsed_problem,
                solution=solution,
                verification=verification,
                explanation=explanation,
                agent_traces=traces
            )

        except Exception as e:
            error_trace = AgentTrace(
                agent_name="System",
                started_at=datetime.now(),
                completed_at=datetime.now(),
                input_summary="Unhandled exception",
                output_summary=str(e),
                status="failed",
                error=str(e)
            )
            traces.append(error_trace)

            return FinalResult(
                status="error",
                raw_input=raw_input,
                agent_traces=traces,
                error_message=str(e)
            )

    # ------------------------------------
    # TRACE HELPERS
    # ------------------------------------
    def _create_trace(self, agent_name: str, input_summary: str) -> AgentTrace:
        return AgentTrace(
            agent_name=agent_name,
            started_at=datetime.now(),
            input_summary=input_summary,
            status="running"
        )

    def _complete_trace(
        self,
        trace: AgentTrace,
        output_summary: str,
        status: str = "completed"
    ):
        trace.completed_at = datetime.now()
        trace.output_summary = output_summary
        trace.status = status
