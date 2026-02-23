"""
Few-Shot Optimizer - Intelligent example selection and ordering.

Selects optimal few-shot examples balancing relevance and diversity.
Uses tuner model from PromptuneConfig via call_llm_structured.
"""

from dataclasses import dataclass

from mcp_servers.utils.llm import call_llm_structured
from schemas import ExampleScores, TrainingExample


@dataclass
class ScoredExample:
    """An example with relevance and diversity scores."""
    example: TrainingExample
    relevance_score: float
    diversity_score: float
    complexity_score: float
    combined_score: float


@dataclass
class SelectionResult:
    """Result of example selection."""
    selected_examples: list[ScoredExample]
    prompt_with_examples: str
    selection_reasoning: str


SCORING_PROMPT = """You are an expert at selecting few-shot examples for prompts.

## BASE PROMPT:
{prompt}

## CANDIDATE EXAMPLES:
{examples}

## YOUR TASK:
Score each example on three dimensions (0.0 to 1.0):

1. **relevance**: How well does this example represent the task described in the prompt?
   - 1.0 = Perfect representation of core task
   - 0.5 = Somewhat related
   - 0.0 = Not relevant

2. **diversity**: How different is this example from typical/common cases?
   - 1.0 = Unique edge case or unusual input
   - 0.5 = Somewhat different
   - 0.0 = Very common/typical case

3. **complexity**: How complex is this example?
   - 1.0 = Very complex (long input, multiple steps, edge cases)
   - 0.5 = Medium complexity
   - 0.0 = Very simple (short, straightforward)

Score ALL examples. Be precise with scores."""


REASONING_PROMPT = """Explain briefly (2-3 sentences) why these examples were selected:

Selected examples:
{selected}

From pool of {total} examples.

Selection strategy: {strategy}

Provide a brief explanation of why this selection is optimal."""


def _format_example_for_prompt(example: TrainingExample, style: str = "markdown") -> str:
    """Format a single example for inclusion in a prompt."""
    if style == "markdown":
        return f"""**Example:**
Input: {example.input}
Output: {example.expected_output}
"""
    elif style == "xml":
        return f"""<example>
<input>{example.input}</input>
<output>{example.expected_output}</output>
</example>
"""
    elif style == "numbered":
        return f"""Input: {example.input}
Output: {example.expected_output}
"""
    elif style == "chat":
        return f"""User: {example.input}
Assistant: {example.expected_output}
"""
    else:
        return f"Input: {example.input}\nOutput: {example.expected_output}\n"


def _estimate_complexity(example: TrainingExample) -> float:
    """Estimate example complexity based on length and structure."""
    input_len = len(example.input)
    output_len = len(example.expected_output)
    total_len = input_len + output_len

    # Normalize to 0-1 scale (assuming max ~500 chars is complex)
    length_score = min(total_len / 500, 1.0)

    # Check for complexity indicators
    complexity_indicators = [
        "\n" in example.input,  # Multi-line
        "\n" in example.expected_output,
        len(example.input.split()) > 10,  # Many words
        any(c in example.input for c in "[]{}()"),  # Code-like
    ]
    indicator_score = sum(complexity_indicators) / len(complexity_indicators)

    return (length_score + indicator_score) / 2


def _compute_diversity_bonus(
    candidate: TrainingExample,
    selected: list[TrainingExample],
) -> float:
    """Compute diversity bonus - higher if candidate is different from selected."""
    if not selected:
        return 0.5  # Neutral bonus for first selection

    # Simple diversity: compare input lengths and first words
    candidate_features = {
        "len": len(candidate.input),
        "first_word": candidate.input.split()[0] if candidate.input.split() else "",
        "has_newline": "\n" in candidate.input,
    }

    max_similarity = 0.0
    for s in selected:
        s_features = {
            "len": len(s.input),
            "first_word": s.input.split()[0] if s.input.split() else "",
            "has_newline": "\n" in s.input,
        }
        # Count matching features
        matches = sum(
            1 for k in candidate_features
            if candidate_features[k] == s_features[k]
        )
        similarity = matches / len(candidate_features)
        max_similarity = max(max_similarity, similarity)

    # Diversity bonus is inverse of similarity
    return 1.0 - max_similarity


async def _score_examples_with_llm(
    prompt: str,
    examples: list[TrainingExample],
    model: str,
) -> list[dict]:
    """Use LLM to score examples for relevance and diversity."""
    examples_text = "\n".join(
        f"[{i}] Input: {e.input}\n    Output: {e.expected_output}"
        for i, e in enumerate(examples)
    )

    scoring_input = SCORING_PROMPT.format(
        prompt=prompt,
        examples=examples_text,
    )

    try:
        result = await call_llm_structured(
            model=model,
            messages=[{"role": "user", "content": scoring_input}],
            response_model=ExampleScores,
            temperature=0.0,
        )
        return [
            {"index": s.index, "relevance": s.relevance, "diversity": s.diversity, "complexity": s.complexity}
            for s in result.scores
        ]
    except Exception:
        # Return neutral scores on failure
        return [
            {"index": i, "relevance": 0.5, "diversity": 0.5, "complexity": 0.5}
            for i in range(len(examples))
        ]


async def select_examples(
    prompt: str,
    example_pool: list[TrainingExample],
    num_examples: int = 3,
    strategy: str = "balanced",
    model: str | None = None,
) -> SelectionResult:
    """
    Select optimal few-shot examples from a pool.

    Args:
        prompt: The base prompt (without examples)
        example_pool: Pool of candidate examples
        num_examples: Number of examples to select
        strategy: Selection strategy (balanced, relevant, diverse, simple_first)
        model: Tuner model to use for scoring (from config)

    Returns:
        SelectionResult with selected examples and formatted prompt
    """
    if not model:
        raise ValueError("Model is required. Pass the tuner model from PromptuneConfig.")

    if len(example_pool) <= num_examples:
        # Just use all examples if pool is small
        scored = [
            ScoredExample(
                example=e,
                relevance_score=1.0,
                diversity_score=1.0,
                complexity_score=_estimate_complexity(e),
                combined_score=1.0,
            )
            for e in example_pool
        ]
        # Sort by complexity for simple_first
        scored.sort(key=lambda x: x.complexity_score)

        examples_text = "\n".join(
            _format_example_for_prompt(s.example) for s in scored
        )

        return SelectionResult(
            selected_examples=scored,
            prompt_with_examples=f"{prompt}\n\n## Examples:\n{examples_text}",
            selection_reasoning="All examples selected (pool smaller than requested count).",
        )

    # Score examples with LLM
    llm_scores = await _score_examples_with_llm(prompt, example_pool, model)

    # Build score lookup
    score_lookup = {s["index"]: s for s in llm_scores}

    # Greedy selection with diversity bonus
    selected: list[ScoredExample] = []
    remaining_indices = set(range(len(example_pool)))

    for _ in range(num_examples):
        best_idx = None
        best_score = -1.0

        for idx in remaining_indices:
            example = example_pool[idx]
            scores = score_lookup.get(idx, {})

            relevance = scores.get("relevance", 0.5)
            llm_diversity = scores.get("diversity", 0.5)
            complexity = scores.get("complexity", _estimate_complexity(example))

            # Compute dynamic diversity bonus
            diversity_bonus = _compute_diversity_bonus(
                example, [s.example for s in selected]
            )

            # Combine scores based on strategy
            if strategy == "relevant":
                combined = relevance * 0.8 + diversity_bonus * 0.2
            elif strategy == "diverse":
                combined = relevance * 0.3 + diversity_bonus * 0.5 + llm_diversity * 0.2
            elif strategy == "simple_first":
                combined = relevance * 0.5 + (1 - complexity) * 0.3 + diversity_bonus * 0.2
            else:  # balanced
                combined = relevance * 0.5 + diversity_bonus * 0.3 + llm_diversity * 0.2

            if combined > best_score:
                best_score = combined
                best_idx = idx

        if best_idx is not None:
            example = example_pool[best_idx]
            scores = score_lookup.get(best_idx, {})
            selected.append(ScoredExample(
                example=example,
                relevance_score=scores.get("relevance", 0.5),
                diversity_score=scores.get("diversity", 0.5),
                complexity_score=scores.get("complexity", _estimate_complexity(example)),
                combined_score=best_score,
            ))
            remaining_indices.remove(best_idx)

    # Sort by complexity if simple_first strategy
    if strategy == "simple_first":
        selected.sort(key=lambda x: x.complexity_score)

    # Format examples into prompt
    examples_text = "\n".join(
        _format_example_for_prompt(s.example) for s in selected
    )

    # Generate reasoning
    reasoning = f"Selected {len(selected)} examples using '{strategy}' strategy. "
    if strategy == "balanced":
        reasoning += "Balanced relevance to task with diversity of cases."
    elif strategy == "relevant":
        reasoning += "Prioritized examples most relevant to the task."
    elif strategy == "diverse":
        reasoning += "Prioritized coverage of different input types."
    elif strategy == "simple_first":
        reasoning += "Ordered from simple to complex for progressive learning."

    return SelectionResult(
        selected_examples=selected,
        prompt_with_examples=f"{prompt}\n\n## Examples:\n{examples_text}",
        selection_reasoning=reasoning,
    )


def format_examples(
    examples: list[TrainingExample],
    format_style: str = "markdown",
) -> str:
    """
    Format examples into a string for prompt inclusion.

    Args:
        examples: List of examples to format
        format_style: Style to use (markdown, xml, numbered, chat)

    Returns:
        Formatted string with all examples
    """
    formatted = []
    for i, ex in enumerate(examples, 1):
        if format_style == "numbered":
            formatted.append(f"{i}. " + _format_example_for_prompt(ex, format_style))
        else:
            formatted.append(_format_example_for_prompt(ex, format_style))

    return "\n".join(formatted)
