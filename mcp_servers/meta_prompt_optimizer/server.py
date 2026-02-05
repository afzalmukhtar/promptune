"""Meta-Prompt Optimizer MCP Server."""

from fastmcp import FastMCP
from mcp_servers.meta_prompt_optimizer.optimizer import optimize

mcp = FastMCP("meta_prompt_optimizer")


@mcp.tool()
async def generate_candidates(prompt: str, feedback: dict, num_candidates: int = 2) -> dict:
    """Generate improved prompt candidates."""
    return await optimize(prompt, feedback, num_candidates)


if __name__ == "__main__":
    mcp.run()
