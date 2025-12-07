"""
Promptune MCP Server.

Single MCP entry point for prompt optimization.
Tools: optimize, optimization_step, evaluate.
"""

from fastmcp import FastMCP

from mcp_servers.evaluator.evaluator import evaluate_prompt
from mcp_servers.orchestrator.orchestrator import (
    OptimizationResult,
    optimize_prompt,
    step,
)
from mcp_servers.utils.config import load_config
from schemas import NegativeTrainingExample, TrainingExample

mcp = FastMCP("promptune")


@mcp.tool()
async def optimize(
    initial_prompt: str,
    training_examples: list[dict] | None = None,
    negative_examples: list[dict] | None = None,
    config_path: str | None = None,
) -> dict:
    """
    Run full optimization on a prompt.

    Supports 3 modes based on which examples are provided:
    - Positive only: standard empirical scoring
    - Negative only: reverse empirical scoring (output must avoid bad patterns)
    - Mixed: combined scoring (average of positive + negative empirical)

    Args:
        initial_prompt: Starting prompt to optimize
        training_examples: List of {input, expected_output} dicts (positive examples)
        negative_examples: List of {input, bad_output, reason_why_bad} dicts (negative examples)
        config_path: Path to promptune.yaml (default: 'promptune.yaml')

    Returns:
        Dict with best_prompt, best_score, iterations, converged, history
    """
    config = load_config(config_path)
    examples = [
        TrainingExample(input=e["input"], expected_output=e["expected_output"])
        for e in (training_examples or [])
    ]
    negatives = [
        NegativeTrainingExample(**e) for e in (negative_examples or [])
    ] or None

    result: OptimizationResult = await optimize_prompt(
        initial_prompt=initial_prompt,
        training_examples=examples,
        config=config,
        negative_examples=negatives,
    )

    return {
        "best_prompt": result.best_prompt,
        "best_score": result.best_score,
        "iterations": result.iterations,
        "converged": result.converged,
        "convergence_reason": result.convergence_reason,
        "history": [
            {
                "iteration": h.iteration,
                "best_score": h.best_score,
                "sample_scores": h.sample_scores,
                "candidates_generated": h.candidates_generated,
                "candidates_evaluated": h.candidates_evaluated,
            }
            for h in result.history
        ],
    }


@mcp.tool()
async def optimization_step(
    candidates: list[str],
    training_examples: list[dict] | None = None,
    negative_examples: list[dict] | None = None,
    config_path: str | None = None,
    optimizers: list[str] | None = None,
) -> dict:
    """
    Run a single optimization step for manual control.

    Args:
        candidates: Current candidate sample of prompts
        training_examples: List of {input, expected_output} dicts (positive examples)
        negative_examples: List of {input, bad_output, reason_why_bad} dicts (negative examples)
        config_path: Path to promptune.yaml (default: 'promptune.yaml')
        optimizers: Which optimizers to use (overrides config)

    Returns:
        Dict with new_candidates, scores, candidates_generated, candidates_evaluated
    """
    config = load_config(config_path)
    examples = [
        TrainingExample(input=e["input"], expected_output=e["expected_output"])
        for e in (training_examples or [])
    ]
    negatives = [
        NegativeTrainingExample(**e) for e in (negative_examples or [])
    ] or None

    return await step(
        candidates=candidates,
        training_examples=examples,
        config=config,
        optimizers=optimizers,
        negative_examples=negatives,
    )


@mcp.tool()
async def evaluate(
    prompt: str,
    training_examples: list[dict] | None = None,
    negative_examples: list[dict] | None = None,
    config_path: str | None = None,
) -> dict:
    """
    Evaluate a single prompt against training examples.

    Use this to score a prompt without running optimization.
    Returns score, strengths, weaknesses, and suggestions.

    Args:
        prompt: The prompt to evaluate
        training_examples: List of {input, expected_output} dicts (positive examples)
        negative_examples: List of {input, bad_output, reason_why_bad} dicts (negative examples)
        config_path: Path to promptune.yaml (default: 'promptune.yaml')

    Returns:
        Dict with score, strengths, weaknesses, suggestions
    """
    config = load_config(config_path)
    examples = [
        TrainingExample(input=e["input"], expected_output=e["expected_output"])
        for e in (training_examples or [])
    ]
    negatives = [
        NegativeTrainingExample(**e) for e in (negative_examples or [])
    ] or None

    result = await evaluate_prompt(
        prompt,
        examples,
        config=config,
        negative_examples=negatives,
    )

    return {
        "score": result.score,
        "strengths": result.strengths,
        "weaknesses": result.weaknesses,
        "suggestions": result.suggestions,
        "feedback": result.feedback,
    }


if __name__ == "__main__":
    mcp.run()
