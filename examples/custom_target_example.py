"""
Example: Using Custom Evaluation Targets with Promptune

This demonstrates how to optimize prompts for black-box systems like:
- Custom retrievers
- RAG pipelines
- Any system where you control the invoke() function

The key: implement EvaluationTarget.invoke(prompt, input) -> output
"""

import asyncio
import logging
import os

from dotenv import load_dotenv
from litellm import acompletion

from mcp_servers.beam_orchestrator.orchestrator import BeamConfig, optimize_beam
from mcp_servers.targets.base import BaseTarget
from schemas import TrainingExample

# Suppress LiteLLM logging noise
os.environ["LITELLM_LOG"] = "ERROR"
for logger_name in ["LiteLLM", "litellm", "httpx", "httpcore"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).propagate = False

load_dotenv()
# =============================================================================
# Example 1: Custom LLM Target with special behavior
# =============================================================================


class CustomLLMTarget(BaseTarget):
    """Example: LLM with custom system behavior."""

    def __init__(self, model: str = "azure/gpt-4.1"):
        self.model = model

    async def invoke(self, prompt: str, input_text: str) -> str:
        """Apply prompt as system message, input as user message."""
        response = await acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    def __repr__(self) -> str:
        return f"CustomLLMTarget(model={self.model!r})"


# =============================================================================
# Example 2: Query Rewriting Target (simulated retriever)
# =============================================================================


class QueryRewriteTarget(BaseTarget):
    """
    Example: Optimize query rewriting prompts for retrieval.

    This simulates a retrieval system:
    1. Use prompt to rewrite the query
    2. "Retrieve" documents (simulated here)
    3. Return retrieved content

    In production, replace _simulate_retrieval with your actual vector DB.
    """

    def __init__(self, model: str = "azure/gpt-4.1"):
        self.model = model
        # Simulated document store
        self.documents = {
            "weather": "Current conditions: Temperature 72°F, Humidity 45%, Sunny skies.",
            "forecast": "5-day forecast: Mon 70°F, Tue 68°F, Wed 75°F, Thu 72°F, Fri 69°F.",
            "temperature": "Temperature readings: Morning 65°F, Afternoon 72°F, Evening 68°F.",
            "python": "Python is a high-level programming language known for its simplicity.",
            "javascript": "JavaScript is the language of the web, running in browsers.",
        }

    async def invoke(self, prompt: str, input_text: str) -> str:
        """
        1. Use prompt to rewrite the query
        2. Retrieve documents matching the rewritten query
        3. Return retrieved content
        """
        # Step 1: Rewrite query using the prompt
        rewrite_response = await acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text},
            ],
            temperature=0.0,
        )
        rewritten_query = rewrite_response.choices[0].message.content or input_text

        # Step 2: Simulate retrieval (replace with your vector DB)
        retrieved = self._simulate_retrieval(rewritten_query)

        return retrieved

    def _simulate_retrieval(self, query: str) -> str:
        """Simulate document retrieval based on keyword matching."""
        query_lower = query.lower()
        results = []
        for keyword, doc in self.documents.items():
            if keyword in query_lower:
                results.append(doc)
        return "\n".join(results) if results else "No relevant documents found."

    def __repr__(self) -> str:
        return "QueryRewriteTarget()"


# =============================================================================
# Demo: Run optimization with custom target
# =============================================================================


async def demo_custom_llm():
    """Demo: Optimize a coding assistant prompt using custom LLM target."""
    print("\n" + "=" * 60)
    print("  DEMO: Custom LLM Target")
    print("=" * 60)

    target = CustomLLMTarget()
    examples = [
        TrainingExample(
            input="Write hello world in Python",
            expected_output="print('Hello, World!')",
        ),
        TrainingExample(
            input="Write a function to add two numbers",
            expected_output="def add(a, b):\n    return a + b",
        ),
    ]

    config = BeamConfig(
        beam_width=2,
        max_iterations=2,
        target_score=0.85,
        optimizers=["meta_prompt"],
    )

    result = await optimize_beam(
        initial_prompt="You are a coding assistant.",
        training_examples=examples,
        config=config,
        target=target,  # <-- Pass custom target
        save_output=True,  # <-- Auto-save to outputs/
    )

    print(f"\nFinal Score: {result.best_score:.0%}")
    print(f"Optimized Prompt:\n{result.best_prompt[:200]}...")


async def demo_query_rewrite():
    """Demo: Optimize a query rewriting prompt for retrieval."""
    print("\n" + "=" * 60)
    print("  DEMO: Query Rewrite Target (Retrieval)")
    print("=" * 60)

    target = QueryRewriteTarget()

    # Training data: original query -> expected retrieved content
    examples = [
        TrainingExample(
            input="What's the weather?",
            expected_output="Current conditions: Temperature 72°F, Humidity 45%, Sunny skies.",
        ),
        TrainingExample(
            input="How hot is it?",
            expected_output="Temperature readings: Morning 65°F, Afternoon 72°F, Evening 68°F.",
        ),
    ]

    config = BeamConfig(
        beam_width=2,
        max_iterations=2,
        target_score=0.80,
        optimizers=["meta_prompt"],
    )

    result = await optimize_beam(
        initial_prompt="Rewrite the user's query to be more specific for search.",
        training_examples=examples,
        config=config,
        target=target,  # <-- Pass retrieval target
    )

    print(f"\nFinal Score: {result.best_score:.0%}")
    print(f"Optimized Prompt:\n{result.best_prompt[:300]}...")


async def main():
    """Run both demos."""
    await demo_custom_llm()
    await demo_query_rewrite()


if __name__ == "__main__":
    asyncio.run(main())
