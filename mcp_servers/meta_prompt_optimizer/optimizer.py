"""
Meta-Prompt Optimizer - LLM-based prompt improvement.

Uses an LLM to analyze feedback and generate targeted prompt improvements.
Now includes prompt understanding injection to target poorly-understood sections.
Uses tuner model from PromptuneConfig via call_llm_structured.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mcp_servers.utils.llm import call_llm_structured
from schemas import OptimizationCandidates, PromptUnderstanding

if TYPE_CHECKING:
    pass


@dataclass
class OptimizedCandidate:
    """A single optimized prompt candidate."""
    prompt: str
    strategy: str
    addressed_weaknesses: list[str]


@dataclass
class OptimizationResult:
    """Result of optimization containing multiple candidates."""
    candidates: list[OptimizedCandidate]
    original_prompt: str


OPTIMIZATION_PROMPT = """You are an expert prompt engineer. Improve the given prompt by implementing the specific suggestions provided.

## ORIGINAL PROMPT:
{original_prompt}

## EVALUATION RESULTS:
Score: {score}/100

### WEAKNESSES TO FIX (required):
{weaknesses}

### SUGGESTIONS TO IMPLEMENT (required):
{suggestions}

### STRENGTHS TO KEEP:
{strengths}

{prompt_understanding_section}

{cross_pollination_section}

## INSTRUCTIONS:
Generate {num_candidates} improved prompt(s). Each MUST:

1. **IMPLEMENT ALL SUGGESTIONS** - Every suggestion listed above must be addressed
2. **FIX ALL WEAKNESSES** - Every weakness must be resolved
3. **KEEP ALL STRENGTHS** - Don't remove what's working
4. **BE COMPLETE** - The output must be a full, ready-to-use prompt

## PROMPT STRUCTURE (include all relevant sections):
```
[ROLE] - Who the AI is
[TASK] - What to do, step by step
[FORMAT] - Exact output format expected
[CONSTRAINTS] - Rules, limitations, boundaries
[EXAMPLES] - At least one input/output example
[ERROR HANDLING] - What to do when things go wrong
```

IMPORTANT: The improved prompt must be significantly longer and more detailed than the original. Include concrete examples."""


CROSS_POLLINATION_SECTION = """
## SUCCESSFUL PROMPTS TO LEARN FROM:
These prompts have scored well. Identify patterns you can adapt:
{prompts}

Look for:
- Structural patterns (how they organize information)
- Phrasing patterns (how they give instructions)
- Component patterns (what sections they include)
"""


PROMPT_UNDERSTANDING_SECTION = """
## PROMPT UNDERSTANDING ANALYSIS:
The following analysis shows how well the target LLM understood each part of the prompt.

### WELL FOLLOWED (reinforce these patterns):
{well_followed}

### POORLY FOLLOWED (fix these — the LLM struggled with them):
{poorly_followed}

Overall compliance: {overall_compliance:.0%}

IMPORTANT: Focus especially on rewriting the poorly-followed sections to be clearer and more explicit.
The well-followed sections are working — keep their patterns but strengthen the weak areas.
"""


def _build_understanding_section(understanding: PromptUnderstanding | None) -> str:
    """Build the prompt understanding section for the optimization prompt."""
    if not understanding:
        return ""

    well_lines = []
    for s in understanding.well_followed:
        well_lines.append(f"- \"{s.section}\" (score: {s.score:.0%}) — Evidence: {s.evidence}")

    poorly_lines = []
    for s in understanding.poorly_followed:
        poorly_lines.append(
            f"- \"{s.section}\" (score: {s.score:.0%}) — Reason: {s.reason}"
        )

    if not well_lines and not poorly_lines:
        return ""

    return PROMPT_UNDERSTANDING_SECTION.format(
        well_followed="\n".join(well_lines) or "None identified",
        poorly_followed="\n".join(poorly_lines) or "None identified",
        overall_compliance=understanding.overall_compliance,
    )


async def optimize(
    prompt: str,
    feedback: dict,
    num_candidates: int = 3,
    cross_pollination_prompts: list[str] | None = None,
    model: str | None = None,
    prompt_understanding: PromptUnderstanding | None = None,
) -> OptimizationResult:
    """
    Generate improved prompt candidates using LLM meta-reasoning.

    Args:
        prompt: The original prompt to improve
        feedback: Evaluation feedback dict with score, weaknesses, suggestions, strengths
        num_candidates: Number of improved variants to generate
        cross_pollination_prompts: Optional list of successful prompts to learn from
        model: Tuner model to use (from config)
        prompt_understanding: Optional analysis of which prompt sections were followed/ignored

    Returns:
        OptimizationResult with list of improved candidates
    """
    if not model:
        raise ValueError("Model is required. Pass the tuner model from PromptuneConfig.")

    # Build cross-pollination section if provided
    cross_section = ""
    if cross_pollination_prompts:
        prompts_text = "\n\n---\n\n".join(
            f"Prompt {i+1}:\n{p}" for i, p in enumerate(cross_pollination_prompts)
        )
        cross_section = CROSS_POLLINATION_SECTION.format(prompts=prompts_text)

    # Build prompt understanding section
    understanding_section = _build_understanding_section(prompt_understanding)

    # Format feedback
    score = feedback.get("score", 0)
    if isinstance(score, float) and score <= 1:
        score = int(score * 100)

    weaknesses = feedback.get("weaknesses", [])
    suggestions = feedback.get("suggestions", [])
    strengths = feedback.get("strengths", [])

    # Build optimization prompt
    opt_prompt = OPTIMIZATION_PROMPT.format(
        original_prompt=prompt,
        score=score,
        weaknesses="\n".join(f"- {w}" for w in weaknesses) or "None identified",
        suggestions="\n".join(f"- {s}" for s in suggestions) or "None provided",
        strengths="\n".join(f"- {s}" for s in strengths) or "None identified",
        cross_pollination_section=cross_section,
        prompt_understanding_section=understanding_section,
        num_candidates=num_candidates,
    )

    # Call LLM with structured output
    result = await call_llm_structured(
        model=model,
        messages=[{"role": "user", "content": opt_prompt}],
        response_model=OptimizationCandidates,
        temperature=0.7,  # Higher temperature for diverse candidates
    )

    # Build candidates from structured response
    candidates = []
    for c in result.candidates:
        candidates.append(OptimizedCandidate(
            prompt=c.prompt,
            strategy=c.strategy,
            addressed_weaknesses=c.addressed_weaknesses,
        ))

    return OptimizationResult(
        candidates=candidates,
        original_prompt=prompt,
    )
