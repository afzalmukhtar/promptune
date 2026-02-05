"""
Meta-Prompt Optimizer - LLM-based prompt improvement.

Uses an LLM to analyze feedback and generate targeted prompt improvements.
"""

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from litellm import acompletion

load_dotenv()


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

## OUTPUT FORMAT:
```json
{{
  "candidates": [
    {{
      "prompt": "The COMPLETE improved prompt with all sections...",
      "strategy": "What improvements were made",
      "addressed_weaknesses": ["list each weakness fixed"],
      "implemented_suggestions": ["list each suggestion implemented"]
    }}
  ]
}}
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


async def optimize(
    prompt: str,
    feedback: dict,
    num_candidates: int = 3,
    cross_pollination_prompts: list[str] | None = None,
    model: str | None = None,
) -> OptimizationResult:
    """
    Generate improved prompt candidates using LLM meta-reasoning.

    Args:
        prompt: The original prompt to improve
        feedback: Evaluation feedback dict with score, weaknesses, suggestions, strengths
        num_candidates: Number of improved variants to generate
        cross_pollination_prompts: Optional list of successful prompts to learn from
        model: Model to use (default: from env config)

    Returns:
        OptimizationResult with list of improved candidates
    """
    model = model or get_default_model()

    # Build cross-pollination section if provided
    cross_section = ""
    if cross_pollination_prompts:
        prompts_text = "\n\n---\n\n".join(
            f"Prompt {i+1}:\n{p}" for i, p in enumerate(cross_pollination_prompts)
        )
        cross_section = CROSS_POLLINATION_SECTION.format(prompts=prompts_text)

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
        num_candidates=num_candidates,
    )

    # Call LLM
    response = await acompletion(
        model=model,
        messages=[{"role": "user", "content": opt_prompt}],
        response_format={"type": "json_object"},
        temperature=0.7,  # Higher temperature for diverse candidates
    )

    # Parse response
    text = response.choices[0].message.content
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Return empty result on parse failure
        return OptimizationResult(candidates=[], original_prompt=prompt)

    # Build candidates
    candidates = []
    for c in data.get("candidates", []):
        candidates.append(OptimizedCandidate(
            prompt=c.get("prompt", ""),
            strategy=c.get("strategy", "Unknown strategy"),
            addressed_weaknesses=c.get("addressed_weaknesses", []),
        ))

    return OptimizationResult(
        candidates=candidates,
        original_prompt=prompt,
    )
