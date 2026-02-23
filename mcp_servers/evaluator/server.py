"""
Evaluator MCP Server.

Provides tools for evaluating prompts against training data using LLM-as-judge.
"""

from fastmcp import FastMCP

from mcp_servers.evaluator.evaluator import evaluate_prompt
from mcp_servers.utils.config import load_config
from schemas import EvaluationResult, TrainingExample

mcp = FastMCP("evaluator")


@mcp.tool()
async def evaluate(
    prompt: str,
    training_examples: list[dict],
    config_path: str | None = None,
) -> dict:
    """
    Evaluate a prompt against training examples.

    Args:
        prompt: The prompt text to evaluate
        training_examples: List of {input, expected_output} pairs
        config_path: Path to promptune.yaml (default: 'promptune.yaml')

    Returns:
        EvaluationResult as dict with score, feedback, strengths, weaknesses
    """
    config = load_config(config_path)
    examples = [TrainingExample(**ex) for ex in training_examples]

    result: EvaluationResult = await evaluate_prompt(
        prompt=prompt,
        training_examples=examples,
        config=config,
    )

    return result.model_dump()


if __name__ == "__main__":
    mcp.run()
