"""
Prompt Test Runner.

Runs a prompt against training examples using the target model and returns
raw outputs for the agent to judge. The agent itself acts as both tuner
and judge — this module only handles target model execution.

Supports custom evaluation targets (black-box systems) via EvaluationTarget protocol.
"""

import asyncio
import hashlib
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mcp_servers.utils.llm import call_llm_plain

if TYPE_CHECKING:
    from mcp_servers.targets.base import EvaluationTarget
    from mcp_servers.utils.config import PromptuneConfig


@dataclass
class TestResult:
    """Result of running a prompt against one example."""

    input: str
    expected_output: str
    actual_output: str


@dataclass
class NegativeTestResult:
    """Result of running a prompt against one negative example."""

    input: str
    bad_output: str
    reason_why_bad: str
    actual_output: str


async def _generate_output(
    prompt: str,
    task_input: str,
    target_model: str,
    target: "EvaluationTarget | None" = None,
) -> str:
    """Generate output using the prompt being evaluated.

    If a target is provided, uses target.invoke(prompt, input).
    Otherwise, uses target_model LLM via plain call (no tool calling).
    """
    if target is not None:
        return await target.invoke(prompt, task_input)

    full_prompt = f"{prompt}\n\n## Input:\n{task_input}"
    return await call_llm_plain(
        model=target_model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.0,
    )


async def run_prompt_tests(
    prompt: str,
    training_examples: list | None = None,
    negative_examples: list | None = None,
    config: "PromptuneConfig | None" = None,
    target_model: str | None = None,
    batch_size: int | None = None,
    target: "EvaluationTarget | None" = None,
) -> dict:
    """
    Run a prompt against training examples and return raw outputs.

    The agent judges the results — this function only runs the target model.

    Args:
        prompt: The prompt text to test
        training_examples: Positive examples with input + expected_output
        negative_examples: Negative examples with input + bad_output + reason_why_bad
        config: PromptuneConfig with target model setting
        target_model: Override target model (or from config)
        batch_size: Number of examples to randomly sample per run
        target: Optional custom evaluation target (black-box system)

    Returns:
        Dict with positive_results and negative_results containing raw outputs
    """
    if config:
        target_model = target_model or config.models.target
        batch_size = batch_size if batch_size is not None else config.optimization.batch_size
    batch_size = batch_size or 5

    if not target_model:
        raise ValueError(
            "Target model not configured. Provide a PromptuneConfig or explicit target_model."
        )

    positive_results: list[TestResult] = []
    negative_results: list[NegativeTestResult] = []

    # Run positive examples
    if training_examples:
        test_examples = training_examples
        if len(training_examples) > batch_size:
            prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 10000
            rng = random.Random(42 + prompt_hash)
            test_examples = rng.sample(training_examples, batch_size)

        tasks = [
            _generate_output(prompt, ex.input, target_model, target)
            for ex in test_examples
        ]
        outputs = await asyncio.gather(*tasks)

        for ex, output in zip(test_examples, outputs):
            positive_results.append(TestResult(
                input=ex.input,
                expected_output=ex.expected_output,
                actual_output=output,
            ))

    # Run negative examples
    if negative_examples:
        test_neg_examples = negative_examples
        if len(negative_examples) > batch_size:
            prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 10000
            rng = random.Random(42 + prompt_hash)
            test_neg_examples = rng.sample(negative_examples, batch_size)

        tasks = [
            _generate_output(prompt, ex.input, target_model, target)
            for ex in test_neg_examples
        ]
        outputs = await asyncio.gather(*tasks)

        for ex, output in zip(test_neg_examples, outputs):
            negative_results.append(NegativeTestResult(
                input=ex.input,
                bad_output=ex.bad_output,
                reason_why_bad=ex.reason_why_bad,
                actual_output=output,
            ))

    return {
        "positive_results": positive_results,
        "negative_results": negative_results,
    }
