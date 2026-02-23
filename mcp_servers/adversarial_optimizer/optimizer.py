"""
Adversarial Optimizer - Hardens prompts against edge cases and failure modes.

Generates adversarial inputs that might break the prompt, then rewrites
the prompt to handle those cases. Uses negative examples if available.
Uses tuner model from PromptuneConfig via call_llm_structured.
"""

from dataclasses import dataclass

from mcp_servers.utils.llm import call_llm_structured
from schemas import (
    AdversarialInputs,
    NegativeTrainingExample,
    OptimizationCandidates,
    PromptUnderstanding,
)


@dataclass
class OptimizedCandidate:
    """A single hardened prompt candidate."""
    prompt: str
    strategy: str
    addressed_weaknesses: list[str]


@dataclass
class OptimizationResult:
    """Result of adversarial optimization."""
    candidates: list[OptimizedCandidate]
    original_prompt: str
    adversarial_cases: list[str]


ADVERSARIAL_GENERATION_PROMPT = """You are a red-team tester trying to break this prompt.

## PROMPT BEING TESTED:
{prompt}

## KNOWN WEAKNESSES:
{weaknesses}

{negative_examples_section}

{understanding_section}

## YOUR TASK:
Generate 3-5 adversarial inputs that would cause this prompt to produce bad output.
Think about:
- Edge cases the prompt doesn't handle
- Ambiguous inputs that could be interpreted multiple ways
- Inputs that exploit missing constraints
- Inputs similar to the known bad examples (if provided)

For each adversarial case, describe the expected failure mode and suggest
how to harden the prompt against it."""


HARDENING_PROMPT = """You are an expert prompt engineer. Harden this prompt against the identified failure modes.

## ORIGINAL PROMPT:
{prompt}

## ADVERSARIAL CASES FOUND:
{adversarial_cases}

## FAILURE MODES:
{failure_modes}

## HARDENING SUGGESTIONS:
{suggestions}

## KNOWN WEAKNESSES:
{weaknesses}

{negative_examples_section}

## INSTRUCTIONS:
Rewrite the prompt to handle ALL the adversarial cases above.
Add explicit constraints, edge case handling, and clarifications.
Keep everything that works well in the original.
Generate 2 hardened variants with different strategies."""


NEGATIVE_EXAMPLES_SECTION = """
## KNOWN BAD OUTPUTS (from user feedback):
These are real examples of bad behavior to guard against:
{examples}
"""


UNDERSTANDING_SECTION = """
## PROMPT COMPLIANCE ANALYSIS:
The LLM poorly followed these sections (focus hardening here):
{poorly_followed}
"""


def _build_negative_examples_section(
    negative_examples: list[NegativeTrainingExample] | None,
) -> str:
    """Build the negative examples section."""
    if not negative_examples:
        return ""
    lines = []
    for i, ex in enumerate(negative_examples[:5], 1):
        lines.append(
            f"{i}. Input: {ex.input}\n"
            f"   Bad Output: {ex.bad_output}\n"
            f"   Why Bad: {ex.reason_why_bad}"
        )
    return NEGATIVE_EXAMPLES_SECTION.format(examples="\n".join(lines))


def _build_understanding_section(understanding: PromptUnderstanding | None) -> str:
    """Build the understanding section for adversarial analysis."""
    if not understanding or not understanding.poorly_followed:
        return ""
    lines = []
    for s in understanding.poorly_followed:
        lines.append(f"- \"{s.section}\" (score: {s.score:.0%}) — {s.reason}")
    return UNDERSTANDING_SECTION.format(poorly_followed="\n".join(lines))


async def optimize(
    prompt: str,
    feedback: dict,
    model: str,
    negative_examples: list[NegativeTrainingExample] | None = None,
    prompt_understanding: PromptUnderstanding | None = None,
) -> OptimizationResult:
    """
    Harden a prompt by generating adversarial inputs and rewriting to handle them.

    Args:
        prompt: The original prompt to harden
        feedback: Evaluation feedback dict with score, weaknesses, suggestions, strengths
        model: Tuner model to use
        negative_examples: Optional known bad output examples
        prompt_understanding: Optional analysis of prompt compliance

    Returns:
        OptimizationResult with hardened candidates and adversarial cases
    """
    if not model:
        raise ValueError("Model is required. Pass the tuner model from PromptuneConfig.")

    weaknesses = feedback.get("weaknesses", [])
    neg_section = _build_negative_examples_section(negative_examples)
    understanding_section = _build_understanding_section(prompt_understanding)

    # Step 1: Generate adversarial inputs
    adv_prompt = ADVERSARIAL_GENERATION_PROMPT.format(
        prompt=prompt,
        weaknesses="\n".join(f"- {w}" for w in weaknesses) or "None identified",
        negative_examples_section=neg_section,
        understanding_section=understanding_section,
    )

    adv_result = await call_llm_structured(
        model=model,
        messages=[{"role": "user", "content": adv_prompt}],
        response_model=AdversarialInputs,
        temperature=0.5,
    )

    # Step 2: Harden the prompt
    hardening_input = HARDENING_PROMPT.format(
        prompt=prompt,
        adversarial_cases="\n".join(f"- {c}" for c in adv_result.adversarial_cases),
        failure_modes="\n".join(f"- {f}" for f in adv_result.failure_modes),
        suggestions="\n".join(f"- {s}" for s in adv_result.hardening_suggestions),
        weaknesses="\n".join(f"- {w}" for w in weaknesses) or "None identified",
        negative_examples_section=neg_section,
    )

    hardened = await call_llm_structured(
        model=model,
        messages=[{"role": "user", "content": hardening_input}],
        response_model=OptimizationCandidates,
        temperature=0.7,
    )

    candidates = [
        OptimizedCandidate(
            prompt=c.prompt,
            strategy=c.strategy,
            addressed_weaknesses=c.addressed_weaknesses,
        )
        for c in hardened.candidates
        if c.prompt
    ]

    return OptimizationResult(
        candidates=candidates,
        original_prompt=prompt,
        adversarial_cases=adv_result.adversarial_cases,
    )
