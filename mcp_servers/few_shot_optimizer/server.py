"""
Few-Shot Optimizer MCP Server.

Provides tools for selecting and formatting optimal few-shot examples.
"""

from fastmcp import FastMCP

from mcp_servers.few_shot_optimizer.optimizer import (
    SelectionResult,
    format_examples,
    select_examples,
)
from mcp_servers.utils.config import load_config
from schemas import TrainingExample

mcp = FastMCP("few_shot_optimizer")


@mcp.tool()
async def select_optimal_examples(
    prompt: str,
    example_pool: list[dict],
    num_examples: int = 3,
    strategy: str = "balanced",
    config_path: str | None = None,
) -> dict:
    """
    Select optimal few-shot examples from a pool.

    Args:
        prompt: The base prompt (without examples)
        example_pool: List of example dicts with 'input' and 'expected_output'
        num_examples: Number of examples to select (default: 3)
        strategy: Selection strategy - balanced, relevant, diverse, simple_first
        config_path: Path to promptune.yaml (default: 'promptune.yaml')

    Returns:
        Dict with selected examples, formatted prompt, and reasoning
    """
    config = load_config(config_path)
    examples = [
        TrainingExample(input=e["input"], expected_output=e["expected_output"])
        for e in example_pool
    ]

    result: SelectionResult = await select_examples(
        prompt=prompt,
        example_pool=examples,
        num_examples=num_examples,
        strategy=strategy,
        model=config.models.tuner,
    )

    return {
        "selected_examples": [
            {
                "input": s.example.input,
                "expected_output": s.example.expected_output,
                "relevance_score": s.relevance_score,
                "diversity_score": s.diversity_score,
                "complexity_score": s.complexity_score,
                "combined_score": s.combined_score,
            }
            for s in result.selected_examples
        ],
        "prompt_with_examples": result.prompt_with_examples,
        "selection_reasoning": result.selection_reasoning,
    }


@mcp.tool()
def format_example_set(
    examples: list[dict],
    format_style: str = "markdown",
) -> str:
    """
    Format examples into a string for prompt inclusion.

    Args:
        examples: List of example dicts with 'input' and 'expected_output'
        format_style: Style - markdown, xml, numbered, chat

    Returns:
        Formatted string with all examples
    """
    training_examples = [
        TrainingExample(input=e["input"], expected_output=e["expected_output"])
        for e in examples
    ]
    return format_examples(training_examples, format_style)


if __name__ == "__main__":
    mcp.run()
