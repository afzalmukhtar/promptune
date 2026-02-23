---
name: clarity_rewriter
description: Rewrites ambiguous prompt sections for clarity. Identifies vague, unclear, or multi-interpretable instructions and replaces them with precise, explicit language. Uses prompt understanding feedback to prioritize sections the LLM struggled with.
---

# Clarity Rewriter

## Purpose

Identify and rewrite ambiguous, vague, or unclear prompt instructions. Focuses on precision — replacing instructions that could be interpreted multiple ways with explicit, unambiguous language.

## How It Works

1. Analyzes the prompt for unclear sentences and instructions
2. Uses prompt understanding feedback (if available) to prioritize sections the LLM struggled with
3. For each unclear sentence, explains WHY it's unclear and provides a clearer version
4. Rewrites the complete prompt with all clarity improvements applied

## When to Use

- **After initial rounds**: When the prompt scores well structurally but outputs are inconsistent
- **With prompt understanding**: When the evaluator shows sections with low compliance scores
- **Ambiguity reduction**: When different inputs produce wildly different output styles
- **Precision tuning**: When the prompt "works" but not reliably

## Two-Step Process

### Step 1: Clarity Analysis

Identifies unclear sentences and explains issues:

```json
{
  "unclear_sentences": [
    "Format the output appropriately",
    "Include relevant details"
  ],
  "rewritten_sentences": [
    "Format the output as a numbered list with one item per line",
    "Include the date, author name, and summary for each entry"
  ],
  "reasoning": [
    "'Appropriately' is vague — could mean JSON, markdown, plain text, etc.",
    "'Relevant details' is undefined — the model must guess which details matter"
  ]
}
```

### Step 2: Prompt Rewrite

Replaces each unclear sentence with its clearer version while preserving all original intent and structure.

## Input

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | Yes | The prompt to rewrite for clarity |
| `model` | Yes | Tuner model from PromptuneConfig |
| `prompt_understanding` | No | PromptUnderstanding from evaluator |

## Output

Returns the rewritten prompt string, or `None` if no improvements are needed.

## What It Targets

| Ambiguity Type | Example | Rewritten |
|---------------|---------|-----------|
| Vague quantifiers | "Include some examples" | "Include exactly 3 examples" |
| Undefined format | "Format nicely" | "Format as markdown with headers" |
| Missing specifics | "Be concise" | "Limit responses to 2-3 sentences" |
| Multi-interpretable | "Handle errors" | "If input is invalid, return 'ERROR: [reason]'" |
| Missing boundaries | "Write a summary" | "Write a 1-sentence summary under 30 words" |

## How It Uses Prompt Understanding

When the evaluator provides section compliance scores:

```json
{
  "poorly_followed": [
    {
      "section": "Output format specification",
      "score": 0.3,
      "reason": "Format instruction was ambiguous — model produced both JSON and plain text"
    }
  ]
}
```

The clarity rewriter **prioritizes** these poorly-followed sections, since low compliance often indicates unclear instructions rather than model inability.

## Configuration

Uses the **tuner** model from `promptune.yaml`:

```yaml
models:
  tuner: "azure/gpt-4o"  # Used for clarity analysis and rewriting
```

Enabled via:

```yaml
optimization:
  optimizers:
    - clarity_rewriter
```
