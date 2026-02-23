"""
Promptune - Optimize Your Prompts

Usage:
    python test_promptune.py                                # Run with defaults
    python test_promptune.py --config myconfig.yaml         # Custom config file
    python test_promptune.py --data examples.json           # Load data from file
    python test_promptune.py --optimizers meta_prompt few_shot adversarial
    python test_promptune.py --batch-size 10 --target 0.95  # Custom batch/target
    python test_promptune.py --converge                     # Run until convergence
    python test_promptune.py --quiet                        # Minimal output
    python test_promptune.py --save                         # Save results to outputs/
"""

import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

from mcp_servers.beam_orchestrator.orchestrator import optimize_beam
from mcp_servers.utils.config import load_config
from mcp_servers.utils.data_loader import load_dataset
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
    parser.add_argument("--config", default=None, help="Config file path (default: promptune.yaml)")
    parser.add_argument("--data", default=None, help="Training data file (JSON or CSV)")
    parser.add_argument(
        "--optimizers", nargs="+", default=None,
        help="Optimizers to use (overrides config). Options: meta_prompt few_shot adversarial example_augmentor clarity_rewriter",
    )
    parser.add_argument("--beam", type=int, default=None, help="Beam width (overrides config)")
    parser.add_argument("--max-iter", type=int, default=None, help="Max iterations (overrides config)")
    parser.add_argument("--target", type=float, default=None, help="Target score 0-1 (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Examples per eval batch (overrides config)")
    parser.add_argument(
        "--converge", action="store_true", help="Run until convergence (20 iterations)"
    )
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--save", nargs="?", const="outputs", default=False, help="Save results (optionally specify dir)")
    args = parser.parse_args()

    # Load config (required)
    config = load_config(args.config)

    # Apply CLI overrides
    if args.beam is not None:
        config.optimization.beam_width = args.beam
    if args.max_iter is not None:
        config.optimization.max_iterations = args.max_iter
    if args.converge:
        config.optimization.max_iterations = 20
    if args.target is not None:
        config.optimization.target_score = args.target
    if args.batch_size is not None:
        config.optimization.batch_size = args.batch_size
    if args.optimizers is not None:
        config.optimization.optimizers = args.optimizers

    # Load training data
    negative_examples = None
    if args.data:
        dataset = load_dataset(args.data)
        examples = dataset.examples
        negative_examples = dataset.negative_examples or None
        if not examples:
            print(f"Warning: No positive examples found in {args.data}")
            print("Using default examples.")
            examples = _default_examples()
    else:
        examples = _default_examples()

    result = await optimize_beam(
        initial_prompt=args.prompt,
        training_examples=examples,
        config=config,
        verbose=not args.quiet,
        save_output=args.save,
        negative_examples=negative_examples,
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


def _default_examples() -> list[TrainingExample]:
    """Default training examples for quick testing."""
    return [
        TrainingExample(
            input="Write hello world in Python",
            expected_output="print('Hello, World!')",
        ),
        TrainingExample(
            input="Write a function to add two numbers",
            expected_output="def add(a, b):\n    return a + b",
        ),
    ]


if __name__ == "__main__":
    asyncio.run(main())
