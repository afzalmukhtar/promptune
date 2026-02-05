"""
Promptune - Optimize Your Prompts

Usage:
    python test_promptune.py                    # Run with defaults
    python test_promptune.py --converge         # Run until convergence
    python test_promptune.py --target 0.90      # Run until 90% score
    python test_promptune.py --quiet            # Minimal output
"""

import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

from mcp_servers.beam_orchestrator.orchestrator import BeamConfig, optimize_beam
from schemas import TrainingExample

# Suppress LiteLLM logging noise
os.environ["LITELLM_LOG"] = "ERROR"
for logger_name in ["LiteLLM", "litellm", "httpx", "httpcore"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).propagate = False
    logging.getLogger(logger_name).handlers = []


load_dotenv()


async def main():
    parser = argparse.ArgumentParser(description="Promptune Prompt Optimizer")
    parser.add_argument("--prompt", default="You are a coding assistant.", help="Initial prompt")
    parser.add_argument("--beam", type=int, default=2, help="Beam width")
    parser.add_argument("--max-iter", type=int, default=5, help="Max iterations")
    parser.add_argument("--target", type=float, default=0.90, help="Target score (0-1)")
    parser.add_argument(
        "--converge", action="store_true", help="Run until convergence (20 iterations)"
    )
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    # Example training data - customize this!
    examples = [
        TrainingExample(
            input="Write hello world in Python",
            expected_output="print('Hello, World!')",
        ),
        TrainingExample(
            input="Write a function to add two numbers",
            expected_output="def add(a, b):\n    return a + b",
        ),
    ]

    if args.converge:
        args.max_iter = 20

    config = BeamConfig(
        beam_width=args.beam,
        max_iterations=args.max_iter,
        target_score=args.target,
        optimizers=["meta_prompt"],
    )

    result = await optimize_beam(
        initial_prompt=args.prompt,
        training_examples=examples,
        config=config,
        verbose=not args.quiet,
    )

    # Final summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  Iterations: {result.iterations}")
    print(f"  Converged: {result.converged} ({result.convergence_reason})")
    print(f"  Final Score: {result.best_score:.0%}")
    print("=" * 60)
    print("\nOPTIMIZED PROMPT:\n")
    print(result.best_prompt)


if __name__ == "__main__":
    asyncio.run(main())
