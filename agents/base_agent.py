"""
Base agent class using Agno framework.
Groq-safe and Gemini-safe.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Union
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.groq import Groq
from config import Config


# -------------------------------------------------
# Model factory (STRICT + FAIL-FAST)
# -------------------------------------------------

def get_model() -> Union[Gemini, Groq]:
    """
    Get the appropriate Agno model based on configuration.
    Fail fast on misconfiguration.
    """
    provider = Config.LLM_PROVIDER.lower()

    if provider == "groq":
        if not Config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is required for Groq provider")

        return Groq(
            id=Config.GROQ_LLM_MODEL,
            api_key=Config.GROQ_API_KEY
        )

    elif provider == "google":
        if not Config.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini provider")

        return Gemini(
            id=Config.GEMINI_MODEL,
            api_key=Config.GOOGLE_API_KEY
        )

    else:
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {Config.LLM_PROVIDER}")


# -------------------------------------------------
# Base Agent
# -------------------------------------------------

class BaseAgent(ABC):
    """
    Base class for all Math Mentor agents using Agno.

    GUARANTEES:
    - Non-solver agents NEVER invoke tools
    - Tool-enabled agents are explicit
    - LLM failures propagate upward (no silent swallowing)
    - run() always returns a string or raises
    """

    def __init__(
        self,
        name: str,
        description: str,
        instructions: list[str],
        tools: Optional[list] = None,
        model: Optional[Union[Gemini, Groq]] = None,
        markdown: bool = False,
        debug_mode: bool = False
    ):
        self.name = name
        self.description = description

        agent_kwargs = {
            "name": name,
            "model": model or get_model(),
            "description": description,
            "instructions": instructions,
            "markdown": markdown,
        }

        # -------------------------------------------------
        # TOOL HANDLING (STRICT + SAFE)
        # -------------------------------------------------
        if tools and isinstance(tools, list) and len(tools) > 0:
            # SolverAgent path → tool calling enabled
            agent_kwargs["tools"] = tools
        else:
            # All other agents → force-disable tools
            agent_kwargs["tools"] = []
            agent_kwargs["tool_choice"] = "none"

        if debug_mode:
            agent_kwargs["verbose"] = True

        self.agent = Agent(**agent_kwargs)

    # -------------------------------------------------
    # Core execution (HARDENED)
    # -------------------------------------------------

    def run(self, message: str) -> str:
        """
        Run the agent with a message.
        Guarantees:
        - Returns non-empty string
        - Raises on failure
        """
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Agent.run() received empty or invalid message")

        try:
            response = self.agent.run(message)
        except Exception as e:
            # Transport / tool / provider failure
            raise RuntimeError(
                f"{self.name} agent execution failed: {str(e)}"
            ) from e

        # Normalize response
        if hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                f"{self.name} agent returned empty response"
            )

        return content

    def run_with_context(self, message: str, context: str) -> str:
        """
        Run the agent with additional context.
        """
        if not context:
            return self.run(message)

        full_message = (
            f"Context:\n{context}\n\n"
            f"---\n\n"
            f"Task:\n{message}"
        )
        return self.run(full_message)

    # -------------------------------------------------
    # Mandatory agent interface
    # -------------------------------------------------

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process input data - must be implemented by subclasses.
        """
        pass
