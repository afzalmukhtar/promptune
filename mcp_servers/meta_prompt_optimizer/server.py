"""
Meta-Prompt Optimizer MCP Server.

Provides tools for generating improved prompt candidates using LLM meta-reasoning.
"""

from fastmcp import FastMCP

from mcp_servers.meta_prompt_optimizer.optimizer import OptimizationResult, optimize

mcp = FastMCP("meta_prompt_optimizer")


@mcp.tool()
async def generate_candidates(
    prompt: str,
    feedback: dict,
    num_candidates: int = 3,
    cross_pollination_prompts: list[str] | None = None,
) -> dict:
    """
    Generate improved prompt candidates from evaluation feedback.

    Args:
        prompt: The current prompt to improve
        feedback: Evaluation feedback with score, weaknesses, suggestions, strengths
        num_candidates: Number of variants to generate (default: 3)
        cross_pollination_prompts: Optional successful prompts to learn from

    Returns:
        Dict with list of candidate prompts and their improvement strategies
    """
    result: OptimizationResult = await optimize(
        prompt=prompt,
        feedback=feedback,
        num_candidates=num_candidates,
        cross_pollination_prompts=cross_pollination_prompts,
    )

    return {
        "original_prompt": result.original_prompt,
        "candidates": [
            {
                "prompt": c.prompt,
                "strategy": c.strategy,
                "addressed_weaknesses": c.addressed_weaknesses,
            }
            for c in result.candidates
        ],
    }


if __name__ == "__main__":
    mcp.run()
