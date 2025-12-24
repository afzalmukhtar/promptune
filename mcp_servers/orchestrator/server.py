"""
Promptune MCP Server.

Single MCP entry point for prompt testing. The agent itself acts as both
tuner (generates improved prompts) and judge (evaluates outputs).

This server only runs the target model against examples and returns raw
outputs for the agent to analyze.

Tool: run_prompt_tests
"""

from fastmcp import FastMCP

from mcp_servers.evaluator.evaluator import run_prompt_tests as _run_prompt_tests
from mcp_servers.utils.config import load_config
from schemas import NegativeTrainingExample, TrainingExample

mcp = FastMCP("promptune")


@mcp.tool()
async def run_prompt_tests(
    prompt: str,
    training_examples: list[dict] | None = None,
    negative_examples: list[dict] | None = None,
    config_path: str | None = None,
) -> dict:
    """
    Run a prompt against training examples using the target model and return raw outputs.

    YOU (the agent) act as the judge — compare actual outputs to expected/bad outputs yourself.
    YOU (the agent) act as the tuner — generate improved prompts yourself based on your analysis.

    This tool ONLY runs the target model and returns what it produced.

    Supports positive examples (input + expected_output) and/or negative examples
    (input + bad_output + reason_why_bad).

    Args:
        prompt: The prompt to test
        training_examples: List of {input, expected_output} dicts (positive examples)
        negative_examples: List of {input, bad_output, reason_why_bad} dicts (negative examples)
        config_path: Path to promptune.yaml (default: 'promptune.yaml')

    Returns:
        Dict with:
        - positive_results: [{input, expected_output, actual_output}, ...]
        - negative_results: [{input, bad_output, reason_why_bad, actual_output}, ...]
    """
    config = load_config(config_path)
    examples = [
        TrainingExample(input=e["input"], expected_output=e["expected_output"])
        for e in (training_examples or [])
    ]
    negatives = [
        NegativeTrainingExample(**e) for e in (negative_examples or [])
    ] or None

    result = await _run_prompt_tests(
        prompt=prompt,
        training_examples=examples,
        negative_examples=negatives,
        config=config,
    )

    # Serialize dataclass results to dicts
    return {
        "positive_results": [
            {
                "input": r.input,
                "expected_output": r.expected_output,
                "actual_output": r.actual_output,
            }
            for r in result["positive_results"]
        ],
        "negative_results": [
            {
                "input": r.input,
                "bad_output": r.bad_output,
                "reason_why_bad": r.reason_why_bad,
                "actual_output": r.actual_output,
            }
            for r in result["negative_results"]
        ],
    }


if __name__ == "__main__":
    mcp.run()
