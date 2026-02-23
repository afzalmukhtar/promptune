"""
Prompt Evaluator with Empirical Output Testing.

Key principle: Actually TEST the prompt against examples, don't just guess.

Evaluation combines:
1. Empirical testing - Run prompt, compare output to expected
2. Structural analysis - Check for required components
3. Adversarial critique - Find hidden weaknesses
4. Prompt understanding - Analyze which sections the LLM followed/ignored

Supports custom evaluation targets (black-box systems) via EvaluationTarget protocol.
Uses separate target/judge models from PromptuneConfig.
"""

import asyncio
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mcp_servers.utils.llm import call_llm_plain, call_llm_structured
from schemas import (
    AdversarialAnalysis,
    EvaluationResult,
    OutputComparison,
    PromptUnderstanding,
    PromptUnderstandingResponse,
    StructuralAnalysis,
    TrainingExample,
)

if TYPE_CHECKING:
    from mcp_servers.targets.base import EvaluationTarget
    from mcp_servers.utils.config import PromptuneConfig


# =============================================================================
# SCORING WEIGHTS (all scoring done in code)
# =============================================================================

WEIGHTS = {
    "empirical": 0.50,  # 50% - actual output quality (the most important!)
    "structural": 0.30,  # 30% - prompt structure
    "adversarial": 0.20,  # 20% - robustness concerns
}


# =============================================================================
# LLM PROMPTS
# =============================================================================

OUTPUT_COMPARISON_PROMPT = """Compare the ACTUAL output to the EXPECTED output for correctness.

## TASK INPUT:
{task_input}

## EXPECTED OUTPUT:
{expected_output}

## ACTUAL OUTPUT:
{actual_output}

## EVALUATION:
Answer these questions:

1. semantic_match: Does the actual output convey the same meaning/accomplish the same goal as expected? (true/false)
2. format_match: Is the format similar (e.g., both code, both prose, similar structure)? (true/false)
3. correctness: Is the actual output factually/functionally correct for the task? (true/false)
4. completeness: Does the actual output fully address the task (not partial)? (true/false)"""


STRUCTURAL_ANALYSIS_PROMPT = """Analyze this prompt's structure. Answer ONLY with true/false for each question.

## PROMPT:
{prompt}

## STRUCTURAL CHECKS:
1. has_role: Does it define WHO the AI should be? (true/false)
2. has_task: Does it explain WHAT to do? (true/false)
3. has_format: Does it specify HOW to format output? (true/false)
4. has_constraints: Does it set boundaries or rules? (true/false)
5. has_examples: Does it include examples? (true/false)"""


ADVERSARIAL_ANALYSIS_PROMPT = """You are a harsh critic. Find problems with this prompt.

## PROMPT:
{prompt}

## YOUR TASK:
1. List 2-3 specific weaknesses (be concrete, cite issues)
2. List 2-3 actionable suggestions (specific fixes)
3. Give a 1-sentence harsh but fair assessment"""


PROMPT_UNDERSTANDING_PROMPT = """Analyze how well the LLM followed the given prompt instructions.

## PROMPT THAT WAS GIVEN:
{prompt}

## TASK INPUT:
{task_input}

## ACTUAL LLM OUTPUT:
{actual_output}

## YOUR TASK:
For each instruction/section in the prompt, determine:
1. Was it well followed? What evidence shows this?
2. Was it poorly followed or ignored? Why?

Return sections that were well followed and sections that were poorly followed.
Each section should have: the section text, evidence from the output, a score (0-1), and reason if poorly followed.
Also give an overall compliance score (0-1)."""


@dataclass
class EmpiricalResult:
    """Result of testing prompt against one example."""

    input: str
    expected: str
    actual: str
    semantic_match: bool = False
    format_match: bool = False
    correctness: bool = False
    completeness: bool = False
    score: int = 0  # 0-100


@dataclass
class EvaluationDetails:
    """Detailed breakdown of evaluation."""

    empirical_score: int = 0
    structural_score: int = 0
    adversarial_score: int = 0
    empirical_results: list[EmpiricalResult] = field(default_factory=list)
    structural_checks: dict = field(default_factory=dict)
    weaknesses: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    assessment: str = ""


def format_examples(examples: list[TrainingExample], max_examples: int = 5) -> str:
    """Format training examples for display."""
    if not examples:
        return "No training examples provided."
    formatted = []
    for i, ex in enumerate(examples[:max_examples], 1):
        formatted.append(f"Example {i}:")
        formatted.append(f"  Input: {ex.input}")
        formatted.append(f"  Expected Output: {ex.expected_output}")
        formatted.append("")
    if len(examples) > max_examples:
        formatted.append(f"... and {len(examples) - max_examples} more examples")
    return "\n".join(formatted)


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


async def _test_single_example(
    prompt: str,
    example: TrainingExample,
    target_model: str,
    judge_model: str,
    target: "EvaluationTarget | None" = None,
) -> EmpiricalResult:
    """Test prompt against a single example and compare output."""
    actual_output = await _generate_output(prompt, example.input, target_model, target)

    comparison_prompt = OUTPUT_COMPARISON_PROMPT.format(
        task_input=example.input,
        expected_output=example.expected_output,
        actual_output=actual_output,
    )
    comparison = await call_llm_structured(
        model=judge_model,
        messages=[{"role": "user", "content": comparison_prompt}],
        response_model=OutputComparison,
        temperature=0.0,
    )

    score = (
        (25 if comparison.semantic_match else 0)
        + (25 if comparison.format_match else 0)
        + (25 if comparison.correctness else 0)
        + (25 if comparison.completeness else 0)
    )

    return EmpiricalResult(
        input=example.input,
        expected=example.expected_output,
        actual=actual_output,
        semantic_match=comparison.semantic_match,
        format_match=comparison.format_match,
        correctness=comparison.correctness,
        completeness=comparison.completeness,
        score=score,
    )


async def _run_empirical_tests(
    prompt: str,
    examples: list[TrainingExample],
    target_model: str,
    judge_model: str,
    batch_size: int = 5,
    iteration: int | None = None,
    target: "EvaluationTarget | None" = None,
) -> tuple[int, list[EmpiricalResult]]:
    """Run prompt against examples and score output quality.

    Uses random batch sampling: each iteration gets a different random subset.
    The seed combines iteration and a prompt hash so different prompts get
    different batches within the same iteration.
    """
    if not examples:
        return 50, []  # Neutral score if no examples

    # Random batch sampling
    if len(examples) <= batch_size:
        test_examples = examples
    else:
        seed = (iteration if iteration is not None else 42) + hash(prompt) % 10000
        rng = random.Random(seed)
        test_examples = rng.sample(examples, batch_size)

    tasks = [
        _test_single_example(prompt, ex, target_model, judge_model, target)
        for ex in test_examples
    ]
    results = await asyncio.gather(*tasks)

    avg_score = round(sum(r.score for r in results) / len(results)) if results else 0
    return avg_score, results


async def _run_structural_analysis(
    prompt: str,
    judge_model: str,
) -> tuple[int, dict]:
    """Analyze prompt structure using structured tool calling."""
    result = await call_llm_structured(
        model=judge_model,
        messages=[{"role": "user", "content": STRUCTURAL_ANALYSIS_PROMPT.format(prompt=prompt)}],
        response_model=StructuralAnalysis,
        temperature=0.0,
    )

    checks = {
        "has_role": result.has_role,
        "has_task": result.has_task,
        "has_format": result.has_format,
        "has_constraints": result.has_constraints,
        "has_examples": result.has_examples,
    }
    score = sum(20 for v in checks.values() if v)
    return score, checks


async def _run_adversarial_analysis(
    prompt: str,
    judge_model: str,
) -> tuple[int, AdversarialAnalysis]:
    """Run adversarial critique using structured tool calling."""
    result = await call_llm_structured(
        model=judge_model,
        messages=[{"role": "user", "content": ADVERSARIAL_ANALYSIS_PROMPT.format(prompt=prompt)}],
        response_model=AdversarialAnalysis,
        temperature=0.0,
    )

    weakness_count = len(result.weaknesses)
    if weakness_count == 0:
        score = 100
    elif weakness_count == 1:
        score = 75
    elif weakness_count == 2:
        score = 50
    else:
        score = 25

    return score, result


async def _analyze_prompt_understanding(
    prompt: str,
    empirical_results: list[EmpiricalResult],
    judge_model: str,
) -> PromptUnderstanding | None:
    """Analyze which parts of the prompt the LLM understood and followed.

    Uses the judge model to compare prompt instructions vs actual outputs.
    """
    if not empirical_results:
        return None

    # Use the first few empirical results for analysis
    analysis_results = empirical_results[:3]

    # Build context from empirical results
    examples_context = ""
    for i, r in enumerate(analysis_results, 1):
        examples_context += (
            f"\n--- Example {i} ---\n"
            f"Input: {r.input}\n"
            f"Actual Output: {r.actual}\n"
        )

    understanding_prompt = PROMPT_UNDERSTANDING_PROMPT.format(
        prompt=prompt,
        task_input=examples_context,
        actual_output="(See outputs above for each example)",
    )

    try:
        raw = await call_llm_structured(
            model=judge_model,
            messages=[{"role": "user", "content": understanding_prompt}],
            response_model=PromptUnderstandingResponse,
            temperature=0.0,
        )

        # Convert typed response to PromptUnderstanding
        return PromptUnderstanding(
            well_followed=raw.well_followed,
            poorly_followed=raw.poorly_followed,
            overall_compliance=raw.overall_compliance,
        )
    except Exception:
        # Prompt understanding is optional — don't fail the whole evaluation
        return None


async def evaluate_prompt(
    prompt: str,
    training_examples: list[TrainingExample],
    config: "PromptuneConfig | None" = None,
    target_model: str | None = None,
    judge_model: str | None = None,
    pass_threshold: int = 70,
    target: "EvaluationTarget | None" = None,
    batch_size: int | None = None,
    iteration: int | None = None,
) -> EvaluationResult:
    """
    Evaluate a prompt using empirical testing + structural analysis.

    Key innovation: Actually RUN the prompt against examples and compare
    outputs, rather than just guessing if it would work.

    Scoring:
    - 50% Empirical (actual output quality)
    - 30% Structural (prompt components)
    - 20% Adversarial (robustness)

    Args:
        prompt: The prompt text to evaluate
        training_examples: Training examples to test against
        config: PromptuneConfig with model settings
        target_model: Override target model (or from config)
        judge_model: Override judge model (or from config)
        pass_threshold: Score threshold to pass (default: 70)
        target: Optional custom evaluation target (black-box system).
                If provided, uses target.invoke(prompt, input) instead of target_model.
        batch_size: Number of examples to randomly sample per evaluation
        iteration: Current iteration number (used as seed for random sampling)

    Returns:
        EvaluationResult with empirically-grounded scores
    """
    # Resolve models from config or explicit params
    if config:
        target_model = target_model or config.models.target
        judge_model = judge_model or config.models.judge
        batch_size = batch_size if batch_size is not None else config.optimization.batch_size
    batch_size = batch_size or 5

    if not target_model or not judge_model:
        raise ValueError(
            "Models not configured. Provide a PromptuneConfig or explicit target_model/judge_model."
        )

    # Run all analyses in parallel for speed
    empirical_task = _run_empirical_tests(
        prompt, training_examples, target_model, judge_model,
        batch_size=batch_size, iteration=iteration, target=target,
    )
    structural_task = _run_structural_analysis(prompt, judge_model)
    adversarial_task = _run_adversarial_analysis(prompt, judge_model)

    (
        (empirical_score, empirical_results),
        (structural_score, structural_checks),
        (adversarial_score, adversarial_data),
    ) = await asyncio.gather(
        empirical_task,
        structural_task,
        adversarial_task,
    )

    # Run prompt understanding analysis (after empirical, uses results)
    prompt_understanding = await _analyze_prompt_understanding(
        prompt, empirical_results, judge_model
    )

    # Compute final score (weighted, all math in code)
    final_score = int(
        empirical_score * WEIGHTS["empirical"]
        + structural_score * WEIGHTS["structural"]
        + adversarial_score * WEIGHTS["adversarial"]
    )

    # Build feedback
    feedback_lines = [
        f"Score: {final_score}/100",
        "",
        f"Empirical (50%): {empirical_score}/100 - Tested against {len(empirical_results)} examples",
        f"Structural (30%): {structural_score}/100 - {sum(structural_checks.values())}/5 checks passed",
        f"Adversarial (20%): {adversarial_score}/100 - {len(adversarial_data.weaknesses)} weaknesses found",
        "",
    ]

    # Add empirical details
    if empirical_results:
        feedback_lines.append("Example Results:")
        for i, r in enumerate(empirical_results, 1):
            status = "✓" if r.score >= 75 else "✗"
            feedback_lines.append(f"  {status} Example {i}: {r.score}/100")
        feedback_lines.append("")

    # Add assessment
    if adversarial_data.assessment:
        feedback_lines.append(f"Assessment: {adversarial_data.assessment}")

    # Add prompt understanding summary
    if prompt_understanding:
        feedback_lines.append("")
        feedback_lines.append(f"Prompt Compliance: {prompt_understanding.overall_compliance:.0%}")
        if prompt_understanding.poorly_followed:
            feedback_lines.append("Poorly followed sections:")
            for section in prompt_understanding.poorly_followed[:3]:
                feedback_lines.append(f"  - {section.section}: {section.reason}")

    # Collect weaknesses
    weaknesses = []
    for check, passed in structural_checks.items():
        if not passed:
            weaknesses.append(f"Missing: {check.replace('_', ' ')}")
    weaknesses.extend(adversarial_data.weaknesses)

    # Collect strengths from passed checks
    strengths = []
    for check, passed in structural_checks.items():
        if passed:
            strengths.append(f"Has {check.replace('_', ' ')}")

    return EvaluationResult(
        prompt=prompt,
        score=final_score / 100,
        passed=final_score >= pass_threshold,
        feedback="\n".join(feedback_lines),
        strengths=strengths[:5],
        weaknesses=weaknesses[:5],
        suggestions=adversarial_data.suggestions,
        clarity_score=structural_score / 100,
        task_alignment_score=empirical_score / 100,
        example_quality_score=adversarial_score / 100,
        prompt_understanding=prompt_understanding,
    )
