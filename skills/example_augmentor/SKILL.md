---
name: example_augmentor
description: Injects positive AND negative behavioral examples into prompts. Creates DO/DON'T sections showing what good and bad output looks like, teaching the model both what to do and what to avoid.
---

# Example Augmentor

## Purpose

Augment prompts with behavioral examples showing both WHAT TO DO and WHAT NOT TO DO. Unlike the few-shot optimizer which only appends positive examples, this optimizer creates structured DO/DON'T sections with explanations.

## How It Works

1. Takes positive training examples (good input/output pairs)
2. Takes negative training examples (bad output + reason why bad)
3. Selects the most illustrative examples of each type
4. Rewrites the prompt to include structured behavioral sections
5. Integrates examples naturally into the prompt flow

## When to Use

- **With negative examples**: When you have data showing bad outputs and why they're bad
- **Teaching boundaries**: When the model needs to understand what NOT to do
- **Reducing common errors**: When there are recurring output mistakes
- **Style enforcement**: When specific tone/format violations need to be prevented

## Input

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | Yes | The prompt to augment |
| `model` | Yes | Tuner model from PromptuneConfig |
| `positive_examples` | No | List of TrainingExample (good pairs) |
| `negative_examples` | No | List of NegativeTrainingExample (bad + reason) |

At least one of `positive_examples` or `negative_examples` must be provided.

## Output

Returns the augmented prompt string with DO/DON'T sections, or `None` if augmentation isn't possible.

## Example

**Before augmentation:**
```
You write professional emails for business contexts.
```

**After augmentation:**
```
You write professional emails for business contexts.

## What GOOD output looks like:
1. Input: Write a follow-up email after an interview
   Expected: "Subject: Thank You - [Position] Interview\n\nDear [Name],\nThank you for..."

## What BAD output looks like — AVOID these patterns:
1. Input: Write a follow-up email after an interview
   Bad: "Hey, just checking if you decided yet?"
   Why Bad: Too casual, no reference to interview details, pressures recipient
```

## Configuration

Uses the **tuner** model from `promptune.yaml`:

```yaml
models:
  tuner: "azure/gpt-4o"  # Used for augmentation rewriting
```

Enabled via:

```yaml
optimization:
  optimizers:
    - example_augmentor
```
