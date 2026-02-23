"""
Beam Orchestrator MCP Server.

Provides tools for running beam search prompt optimization.
"""

from fastmcp import FastMCP

from mcp_servers.beam_orchestrator.orchestrator import (
    OptimizationResult,
    optimize_beam,
    step,
)
from mcp_servers.utils.config import load_config
from schemas import TrainingExample

mcp = FastMCP("beam_orchestrator")


@mcp.tool()
async def optimize(
    initial_prompt: str,
    training_examples: list[dict],
    config_path: str | None = None,
) -> dict:
    """
    Run full beam search optimization on a prompt.

    Args:
        initial_prompt: Starting prompt to optimize
        training_examples: List of example dicts with 'input' and 'expected_output'
        config_path: Path to promptune.yaml (default: 'promptune.yaml')

    Returns:
        Dict with best_prompt, best_score, iterations, converged, history
    """
    config = load_config(config_path)
    examples = [
        TrainingExample(input=e["input"], expected_output=e["expected_output"])
        for e in training_examples
    ]

    result: OptimizationResult = await optimize_beam(
        initial_prompt=initial_prompt,
        training_examples=examples,
        config=config,
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
                "beam_scores": h.beam_scores,
                "candidates_generated": h.candidates_generated,
                "candidates_evaluated": h.candidates_evaluated,
            }
            for h in result.history
        ],
    }


@mcp.tool()
async def optimization_step(
    beam: list[str],
    training_examples: list[dict],
    config_path: str | None = None,
    optimizers: list[str] | None = None,
) -> dict:
    """
    Run a single optimization step for manual control.

    Args:
        beam: Current beam of prompts
        training_examples: List of example dicts
        config_path: Path to promptune.yaml (default: 'promptune.yaml')
        optimizers: Which optimizers to use (overrides config)

    Returns:
        Dict with new_beam, scores, candidates_generated, candidates_evaluated
    """
    config = load_config(config_path)
    examples = [
        TrainingExample(input=e["input"], expected_output=e["expected_output"])
        for e in training_examples
    ]

    return await step(
        beam=beam,
        training_examples=examples,
        config=config,
        optimizers=optimizers,
    )


if __name__ == "__main__":
    mcp.run()
