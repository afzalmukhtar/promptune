"""
Beam Orchestrator MCP Server.

Provides tools for running beam search prompt optimization.
"""

from fastmcp import FastMCP

from mcp_servers.beam_orchestrator.orchestrator import (
    BeamConfig,
    OptimizationResult,
    optimize_beam,
    step,
)
from schemas import TrainingExample

mcp = FastMCP("beam_orchestrator")


@mcp.tool()
async def optimize(
    initial_prompt: str,
    training_examples: list[dict],
    config: dict | None = None,
) -> dict:
    """
    Run full beam search optimization on a prompt.

    Args:
        initial_prompt: Starting prompt to optimize
        training_examples: List of example dicts with 'input' and 'expected_output'
        config: Optional config with beam_width, max_iterations, target_score, etc.

    Returns:
        Dict with best_prompt, best_score, iterations, converged, history
    """
    examples = [
        TrainingExample(input=e["input"], expected_output=e["expected_output"])
        for e in training_examples
    ]

    beam_config = None
    if config:
        beam_config = BeamConfig(
            beam_width=config.get("beam_width", 3),
            max_iterations=config.get("max_iterations", 10),
            target_score=config.get("target_score", 0.90),
            convergence_threshold=config.get("convergence_threshold", 0.02),
            convergence_patience=config.get("convergence_patience", 3),
            optimizers=config.get("optimizers", ["meta_prompt", "few_shot"]),
        )

    result: OptimizationResult = await optimize_beam(
        initial_prompt=initial_prompt,
        training_examples=examples,
        config=beam_config,
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
    optimizers: list[str] | None = None,
) -> dict:
    """
    Run a single optimization step for manual control.

    Args:
        beam: Current beam of prompts
        training_examples: List of example dicts
        optimizers: Which optimizers to use (default: meta_prompt, few_shot)

    Returns:
        Dict with new_beam, scores, candidates_generated, candidates_evaluated
    """
    examples = [
        TrainingExample(input=e["input"], expected_output=e["expected_output"])
        for e in training_examples
    ]

    return await step(
        beam=beam,
        training_examples=examples,
        optimizers=optimizers,
    )


if __name__ == "__main__":
    mcp.run()
