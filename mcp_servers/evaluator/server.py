"""
Evaluator MCP Server.

Provides tools for evaluating prompts against training data using LLM-as-judge.
"""

from fastmcp import FastMCP

from mcp_servers.evaluator.evaluator import evaluate_prompt
from schemas import EvaluationResult, TrainingExample

mcp = FastMCP("evaluator")


@mcp.tool()
async def evaluate(
    prompt: str,
    training_examples: list[dict],
    model: str | None = None,
) -> dict:
    """
    Evaluate a prompt against training examples.

    Args:
        prompt: The prompt text to evaluate
        training_examples: List of {input, expected_output} pairs
        model: Optional model override (default: uses env config)

    Returns:
        EvaluationResult as dict with score, feedback, strengths, weaknesses
    """
    # Convert dicts to TrainingExample models
    examples = [TrainingExample(**ex) for ex in training_examples]

    # Call core evaluation logic
    result: EvaluationResult = await evaluate_prompt(
        prompt=prompt,
        training_examples=examples,
        model=model,
    )

    # Return as dict for MCP serialization
    return result.model_dump()


if __name__ == "__main__":
    mcp.run()
