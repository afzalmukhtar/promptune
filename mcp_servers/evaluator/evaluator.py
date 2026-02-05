"""
Prompt Evaluator with Empirical Output Testing.

Key principle: Actually TEST the prompt against examples, don't just guess.

Evaluation combines:
1. Empirical testing - Run prompt, compare output to expected
2. Structural analysis - Check for required components
3. Adversarial critique - Find hidden weaknesses

Supports custom evaluation targets (black-box systems) via EvaluationTarget protocol.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from litellm import acompletion

from schemas import EvaluationResult, TrainingExample

if TYPE_CHECKING:
    from mcp_servers.targets.base import EvaluationTarget

load_dotenv()


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
4. completeness: Does the actual output fully address the task (not partial)? (true/false)

## OUTPUT (JSON only):
```json
{{
  "semantic_match": true/false,
  "format_match": true/false,
  "correctness": true/false,
  "completeness": true/false,
  "explanation": "Brief explanation of differences if any"
}}
```"""


STRUCTURAL_ANALYSIS_PROMPT = """Analyze this prompt's structure. Answer ONLY with true/false for each question.

## PROMPT:
{prompt}

## STRUCTURAL CHECKS:
1. has_role: Does it define WHO the AI should be? (true/false)
2. has_task: Does it explain WHAT to do? (true/false)
3. has_format: Does it specify HOW to format output? (true/false)
4. has_constraints: Does it set boundaries or rules? (true/false)
5. has_examples: Does it include examples? (true/false)

## OUTPUT (JSON only):
```json
{{
  "has_role": true/false,
  "has_task": true/false,
  "has_format": true/false,
  "has_constraints": true/false,
  "has_examples": true/false
}}
```"""


ADVERSARIAL_ANALYSIS_PROMPT = """You are a harsh critic. Find problems with this prompt.

## PROMPT:
{prompt}

## YOUR TASK:
1. List 2-3 specific weaknesses (be concrete, cite issues)
2. List 2-3 actionable suggestions (specific fixes)
3. Give a 1-sentence harsh but fair assessment

## OUTPUT (JSON only):
```json
{{
  "weaknesses": ["specific weakness 1", "specific weakness 2"],
  "suggestions": ["specific fix 1", "specific fix 2"],
  "assessment": "One sentence harsh assessment"
}}
```"""


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


def get_default_model() -> str:
    """Get the default model from environment variables."""
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_model = os.getenv("AZURE_OPENAI_MODEL")
    if azure_endpoint and azure_model:
        return f"azure/{azure_model}"
    ollama_base = os.getenv("OLLAMA_API_BASE")
    if ollama_base:
        return "ollama/llama3.2"
    return "gpt-4o-mini"


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


async def _call_llm(prompt: str, model: str, json_mode: bool = True) -> dict | str:
    """Call LLM and optionally parse JSON response."""
    response = await acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"} if json_mode else None,
        temperature=0.0,
    )
    text = response.choices[0].message.content
    if json_mode:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
    return text


async def _generate_output(
    prompt: str,
    task_input: str,
    model: str,
    target: "EvaluationTarget | None" = None,
) -> str:
    """Generate output using the prompt being evaluated.

    If a target is provided, uses target.invoke(prompt, input).
    Otherwise, uses default LLM behavior.
    """
    if target is not None:
        # Use user's black-box target
        return await target.invoke(prompt, task_input)

    # Default: use LLM directly
    full_prompt = f"{prompt}\n\n## Input:\n{task_input}"
    response = await acompletion(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content


async def _test_single_example(
    prompt: str,
    example: TrainingExample,
    model: str,
    target: "EvaluationTarget | None" = None,
) -> EmpiricalResult:
    """Test prompt against a single example and compare output."""
    # Generate actual output using target or default LLM
    actual_output = await _generate_output(prompt, example.input, model, target)

    # Compare to expected
    comparison_prompt = OUTPUT_COMPARISON_PROMPT.format(
        task_input=example.input,
        expected_output=example.expected_output,
        actual_output=actual_output,
    )
    comparison = await _call_llm(comparison_prompt, model)

    # Calculate score from binary checks (25 pts each)
    semantic = comparison.get("semantic_match", False)
    format_ok = comparison.get("format_match", False)
    correct = comparison.get("correctness", False)
    complete = comparison.get("completeness", False)

    score = (
        (25 if semantic else 0)
        + (25 if format_ok else 0)
        + (25 if correct else 0)
        + (25 if complete else 0)
    )

    return EmpiricalResult(
        input=example.input,
        expected=example.expected_output,
        actual=actual_output,
        semantic_match=semantic,
        format_match=format_ok,
        correctness=correct,
        completeness=complete,
        score=score,
    )


async def _run_empirical_tests(
    prompt: str,
    examples: list[TrainingExample],
    model: str,
    max_examples: int = 3,
    target: "EvaluationTarget | None" = None,
) -> tuple[int, list[EmpiricalResult]]:
    """Run prompt against examples and score output quality."""
    if not examples:
        return 50, []  # Neutral score if no examples

    # Test up to max_examples in parallel
    test_examples = examples[:max_examples]
    tasks = [_test_single_example(prompt, ex, model, target) for ex in test_examples]
    results = await asyncio.gather(*tasks)

    # Average score across examples
    avg_score = sum(r.score for r in results) // len(results) if results else 0
    return avg_score, results


async def _run_structural_analysis(prompt: str, model: str) -> tuple[int, dict]:
    """Analyze prompt structure."""
    result = await _call_llm(
        STRUCTURAL_ANALYSIS_PROMPT.format(prompt=prompt),
        model,
    )

    # Score: 20 pts per check (5 checks = 100 max)
    checks = {
        "has_role": result.get("has_role", False),
        "has_task": result.get("has_task", False),
        "has_format": result.get("has_format", False),
        "has_constraints": result.get("has_constraints", False),
        "has_examples": result.get("has_examples", False),
    }
    score = sum(20 for v in checks.values() if v)
    return score, checks


async def _run_adversarial_analysis(prompt: str, model: str) -> tuple[int, dict]:
    """Run adversarial critique."""
    result = await _call_llm(
        ADVERSARIAL_ANALYSIS_PROMPT.format(prompt=prompt),
        model,
    )

    weaknesses = result.get("weaknesses", [])
    suggestions = result.get("suggestions", [])
    assessment = result.get("assessment", "")

    # Score based on number of weaknesses found (fewer = better)
    # 0 weaknesses = 100, 1 = 75, 2 = 50, 3+ = 25
    weakness_count = len(weaknesses)
    if weakness_count == 0:
        score = 100
    elif weakness_count == 1:
        score = 75
    elif weakness_count == 2:
        score = 50
    else:
        score = 25

    return score, {
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "assessment": assessment,
    }


async def evaluate_prompt(
    prompt: str,
    training_examples: list[TrainingExample],
    model: str | None = None,
    pass_threshold: int = 70,
    target: "EvaluationTarget | None" = None,
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
        model: Model to use (default: from env config)
        pass_threshold: Score threshold to pass (default: 70)
        target: Optional custom evaluation target (black-box system).
                If provided, uses target.invoke(prompt, input) instead of LLM.

    Returns:
        EvaluationResult with empirically-grounded scores
    """
    model = model or get_default_model()

    # Run all analyses in parallel for speed
    empirical_task = _run_empirical_tests(prompt, training_examples, model, target=target)
    structural_task = _run_structural_analysis(prompt, model)
    adversarial_task = _run_adversarial_analysis(prompt, model)

    (
        (empirical_score, empirical_results),
        (structural_score, structural_checks),
        (adversarial_score, adversarial_data),
    ) = await asyncio.gather(
        empirical_task,
        structural_task,
        adversarial_task,
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
        f"Adversarial (20%): {adversarial_score}/100 - {len(adversarial_data.get('weaknesses', []))} weaknesses found",
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
    if adversarial_data.get("assessment"):
        feedback_lines.append(f"Assessment: {adversarial_data['assessment']}")

    # Collect weaknesses
    weaknesses = []
    # Failed structural checks
    for check, passed in structural_checks.items():
        if not passed:
            weaknesses.append(f"Missing: {check.replace('_', ' ')}")
    # Adversarial weaknesses
    weaknesses.extend(adversarial_data.get("weaknesses", []))

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
        suggestions=adversarial_data.get("suggestions", []),
        clarity_score=structural_score / 100,
        task_alignment_score=empirical_score / 100,
        example_quality_score=adversarial_score / 100,
    )
