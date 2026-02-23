---
name: adversarial_optimizer
description: Hardens prompts against edge cases and failure modes. Generates adversarial inputs that might break the prompt, then rewrites it to handle those cases. Uses negative examples and prompt understanding analysis.
---

# Adversarial Optimizer

## Purpose

Harden prompts against edge cases, ambiguous inputs, and adversarial attacks. Instead of just improving what works, this optimizer finds what BREAKS and fixes it.

## How It Works

1. Analyzes the prompt for potential failure modes
2. Generates adversarial inputs designed to break the prompt
3. Uses negative training examples (if available) to understand known failure patterns
4. Uses prompt understanding analysis to target poorly-followed sections
5. Rewrites the prompt to handle discovered edge cases

## When to Use

- **After initial optimization**: When the prompt works for normal cases but fails on edge cases
- **With negative examples**: When you have data showing what bad output looks like
- **For robustness**: When reliability matters more than peak performance
- **Production hardening**: Before deploying a prompt to handle diverse real-world inputs

## Input

The optimizer accepts:

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | Yes | The prompt to harden |
| `feedback` | Yes | Evaluation feedback dict (score, weaknesses, suggestions) |
| `model` | Yes | Tuner model from PromptuneConfig |
| `num_candidates` | No | Number of hardened variants (default: 2) |
| `negative_examples` | No | List of NegativeTrainingExample showing bad outputs |
| `prompt_understanding` | No | PromptUnderstanding analysis from evaluator |

## Output

Returns `OptimizationCandidates` with hardened prompt variants:

```json
{
  "candidates": [
    {
      "prompt": "Hardened prompt with edge case handling...",
      "strategy": "Added boundary checks and error handling for empty inputs",
      "addressed_weaknesses": ["No edge case handling", "Fails on empty input"]
    }
  ]
}
```

## How It Uses Negative Examples

When negative examples are provided:

```json
{
  "sample_prompt": "You write professional emails.",
  "input": "Write a follow-up email",
  "bad_output": "Hey, just checking if you decided yet?",
  "reason_why_bad": "Too casual, no reference to interview details"
}
```

The optimizer:
1. Identifies the pattern that led to bad output
2. Adds explicit instructions to avoid that pattern
3. Includes guardrails in the rewritten prompt

## How It Uses Prompt Understanding

When the evaluator provides section compliance data:

```json
{
  "poorly_followed": [
    {"section": "Output format", "score": 0.3, "reason": "Format spec was ambiguous"}
  ]
}
```

The optimizer specifically targets these poorly-followed sections for hardening.

## Configuration

Uses the **tuner** model from `promptune.yaml`:

```yaml
models:
  tuner: "azure/gpt-4o"  # Used for adversarial generation
```

The optimizer is enabled in the orchestrator via:

```yaml
optimization:
  optimizers:
    - adversarial
```
