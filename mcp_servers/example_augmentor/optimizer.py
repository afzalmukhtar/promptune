"""
Example Augmentor - Injects positive AND negative examples into prompts.

Unlike the few-shot optimizer which only appends positive examples,
this optimizer adds DO/DON'T sections with both positive and negative
examples, including reason_why_bad explanations.

Uses tuner model from PromptuneConfig via call_llm_structured.
"""

from mcp_servers.utils.llm import call_llm_structured
from schemas import (
    CandidateOutput,
    NegativeTrainingExample,
    TrainingExample,
)

AUGMENTATION_PROMPT = """You are an expert prompt engineer. Augment this prompt with behavioral examples.

## ORIGINAL PROMPT:
{prompt}

## POSITIVE EXAMPLES (what GOOD output looks like):
{positive_examples}

## NEGATIVE EXAMPLES (what BAD output looks like and WHY):
{negative_examples}

## INSTRUCTIONS:
Rewrite the prompt to include:
1. A "What GOOD output looks like" section with the best positive examples
2. A "What BAD output looks like — AVOID these patterns" section with negative examples and reasons
3. Keep ALL original instructions intact
4. The examples should be naturally integrated, not just appended

The goal is to teach the model both WHAT to do and WHAT NOT to do.
Output ONLY the complete rewritten prompt."""


async def augment(
    prompt: str,
    positive_examples: list[TrainingExample] | None = None,
    negative_examples: list[NegativeTrainingExample] | None = None,
    model: str | None = None,
) -> str | None:
    """
    Augment a prompt with positive and negative behavioral examples.

    Args:
        prompt: The original prompt to augment
        positive_examples: Good input/output examples
        negative_examples: Bad output examples with reasons
        model: Tuner model to use

    Returns:
        Augmented prompt string, or None if augmentation not possible
    """
    if not model:
        raise ValueError("Model is required. Pass the tuner model from PromptuneConfig.")

    # Need at least some examples to augment
    if not positive_examples and not negative_examples:
        return None

    # Format positive examples
    pos_text = "None provided."
    if positive_examples:
        pos_lines = []
        for i, ex in enumerate(positive_examples[:5], 1):
            pos_lines.append(
                f"{i}. Input: {ex.input}\n"
                f"   Expected Output: {ex.expected_output}"
            )
        pos_text = "\n".join(pos_lines)

    # Format negative examples
    neg_text = "None provided."
    if negative_examples:
        neg_lines = []
        for i, ex in enumerate(negative_examples[:5], 1):
            neg_lines.append(
                f"{i}. Input: {ex.input}\n"
                f"   Bad Output: {ex.bad_output}\n"
                f"   Why Bad: {ex.reason_why_bad}"
            )
        neg_text = "\n".join(neg_lines)

    aug_prompt = AUGMENTATION_PROMPT.format(
        prompt=prompt,
        positive_examples=pos_text,
        negative_examples=neg_text,
    )

    try:
        result = await call_llm_structured(
            model=model,
            messages=[{"role": "user", "content": aug_prompt}],
            response_model=CandidateOutput,
            temperature=0.5,
        )
        return result.prompt if result.prompt else None
    except Exception:
        return None
