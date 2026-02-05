"""Evaluator MCP Server."""

from fastmcp import FastMCP
from mcp_servers.evaluator.evaluator import evaluate_prompt
from schemas import TrainingExample, EvaluationResult

mcp = FastMCP("evaluator")


@mcp.tool()
async def evaluate(prompt: str, examples: list[dict]) -> dict:
    """Evaluate a prompt against training examples."""
    training_examples = [
        TrainingExample(input=e["input"], expected_output=e["expected_output"])
        for e in examples
    ]
    return await evaluate_prompt(prompt, training_examples)


if __name__ == "__main__":
    mcp.run()
