"""
Evaluator MCP Server.

Provides tools for evaluating prompts against training data using LLM-as-judge.
"""

from fastmcp import FastMCP

from mcp_servers.evaluator.evaluator import evaluate_prompt
from mcp_servers.utils.config import load_config
from schemas import EvaluationResult, NegativeTrainingExample, TrainingExample

mcp = FastMCP("evaluator")


@mcp.tool()
async def evaluate(
    prompt: str,
    training_examples: list[dict] | None = None,
    negative_examples: list[dict] | None = None,
    config_path: str | None = None,
) -> dict:
    """
    Evaluate a prompt against training examples.

    Supports 3 modes based on which examples are provided:
    - Positive only: standard empirical scoring (output matches expected)
    - Negative only: reverse empirical scoring (output must avoid bad patterns)
    - Mixed: combined scoring (average of positive + negative empirical)

    Args:
        prompt: The prompt text to evaluate
        training_examples: List of {input, expected_output} dicts (positive examples)
        negative_examples: List of {input, bad_output, reason_why_bad} dicts (negative examples)
        config_path: Path to promptune.yaml (default: 'promptune.yaml')

    Returns:
        EvaluationResult as dict with score, feedback, strengths, weaknesses
    """
    config = load_config(config_path)
    examples = [TrainingExample(**ex) for ex in (training_examples or [])]
    negatives = [
        NegativeTrainingExample(**ex) for ex in (negative_examples or [])
    ] or None

    result: EvaluationResult = await evaluate_prompt(
        prompt=prompt,
        training_examples=examples,
        config=config,
        negative_examples=negatives,
    )

    return result.model_dump()


if __name__ == "__main__":
    mcp.run()
