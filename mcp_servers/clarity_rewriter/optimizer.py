"""
Clarity Rewriter - Rewrites ambiguous prompt sections for clarity.

Identifies unclear, ambiguous, or vague sentences in a prompt and rewrites
them to be more explicit and precise. Uses prompt understanding feedback
to target sections the LLM struggled with.

Uses tuner model from PromptuneConfig via call_llm_structured.
"""

from mcp_servers.utils.llm import call_llm_structured
from schemas import ClarityAnalysis, PromptUnderstanding

CLARITY_ANALYSIS_PROMPT = """You are an expert at identifying ambiguity in prompts.

## PROMPT TO ANALYZE:
{prompt}

{understanding_section}

## YOUR TASK:
Find sentences or instructions that are ambiguous, vague, or unclear.
For each unclear sentence:
1. Quote the exact unclear sentence
2. Explain WHY it's unclear (what could be misinterpreted?)
3. Provide a clearer rewritten version

Focus especially on:
- Instructions that could be interpreted multiple ways
- Vague quantifiers ("some", "a few", "appropriate")
- Missing specifics (format details, edge cases, boundaries)
- Sections the LLM previously struggled with (if understanding analysis is provided)"""


REWRITE_PROMPT = """You are an expert prompt engineer. Rewrite this prompt for maximum clarity.

## ORIGINAL PROMPT:
{prompt}

## UNCLEAR SECTIONS FOUND:
{unclear_sections}

## CLEARER REPLACEMENTS:
{replacements}

## REASONING:
{reasoning}

## INSTRUCTIONS:
Rewrite the COMPLETE prompt, replacing each unclear section with its clearer version.
Also improve any other minor clarity issues you notice.
Keep ALL original intent and structure — only improve clarity.
Output the complete rewritten prompt."""


UNDERSTANDING_SECTION = """
## PROMPT UNDERSTANDING ANALYSIS:
The LLM struggled with these sections (prioritize clarifying these):
{poorly_followed}
"""


def _build_understanding_section(understanding: PromptUnderstanding | None) -> str:
    """Build the understanding section for clarity analysis."""
    if not understanding or not understanding.poorly_followed:
        return ""
    lines = []
    for s in understanding.poorly_followed:
        lines.append(f'- "{s.section}" (compliance: {s.score:.0%}) — {s.reason}')
    return UNDERSTANDING_SECTION.format(poorly_followed="\n".join(lines))


async def rewrite(
    prompt: str,
    model: str | None = None,
    prompt_understanding: PromptUnderstanding | None = None,
) -> str | None:
    """
    Rewrite a prompt for improved clarity.

    Args:
        prompt: The original prompt to rewrite
        model: Tuner model to use
        prompt_understanding: Optional analysis of which sections the LLM struggled with

    Returns:
        Rewritten prompt string, or None if no improvements needed
    """
    if not model:
        raise ValueError("Model is required. Pass the tuner model from PromptuneConfig.")

    understanding_section = _build_understanding_section(prompt_understanding)

    # Step 1: Identify unclear sections
    analysis_prompt = CLARITY_ANALYSIS_PROMPT.format(
        prompt=prompt,
        understanding_section=understanding_section,
    )

    try:
        analysis = await call_llm_structured(
            model=model,
            messages=[{"role": "user", "content": analysis_prompt}],
            response_model=ClarityAnalysis,
            temperature=0.3,
        )
    except Exception:
        return None

    # If nothing unclear found, no rewrite needed
    if not analysis.unclear_sentences:
        return None

    # Step 2: Rewrite the prompt with clearer versions
    unclear_text = "\n".join(f'- "{s}"' for s in analysis.unclear_sentences)
    replacements_text = "\n".join(f'- "{s}"' for s in analysis.rewritten_sentences)
    reasoning_text = "\n".join(f"- {r}" for r in analysis.reasoning)

    rewrite_input = REWRITE_PROMPT.format(
        prompt=prompt,
        unclear_sections=unclear_text,
        replacements=replacements_text,
        reasoning=reasoning_text,
    )

    try:
        from schemas import CandidateOutput

        result = await call_llm_structured(
            model=model,
            messages=[{"role": "user", "content": rewrite_input}],
            response_model=CandidateOutput,
            temperature=0.3,
        )
        return result.prompt if result.prompt else None
    except Exception:
        return None
