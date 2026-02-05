"""Few-Shot Optimizer MCP Server."""

from fastmcp import FastMCP
from mcp_servers.few_shot_optimizer.optimizer import select_examples

mcp = FastMCP("few_shot_optimizer")


@mcp.tool()
async def select_optimal_examples(prompt: str, example_pool: list[dict], num_examples: int = 3) -> dict:
    """Select optimal few-shot examples."""
    return await select_examples(prompt, example_pool, num_examples)


if __name__ == "__main__":
    mcp.run()
