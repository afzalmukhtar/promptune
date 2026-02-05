---
name: meta_prompt_optimizer
description: Generates improved prompt candidates using LLM-based meta-reasoning. Analyzes evaluation feedback and creates targeted improvements. Use when you need to optimize a prompt based on its weaknesses.
---

# Meta-Prompt Optimizer

## Purpose

Generate improved prompt candidates by using an LLM to analyze evaluation feedback and apply targeted improvements. This is the "thinking about prompts" optimizer.

## How It Works

1. Takes a prompt and its evaluation feedback
2. Analyzes weaknesses and suggestions
3. Generates improved variants addressing specific issues
4. Can incorporate insights from other successful prompts (cross-pollination)

## When to Use

- **After evaluation**: When you have feedback about a prompt's weaknesses
- **Targeted improvement**: When you know what specific issues to fix
- **Creative variations**: When you need diverse improvement strategies
- **Cross-pollination**: When you want to combine strengths from multiple prompts

## MCP Tools Available

### optimize

Generate improved prompt candidates from a base prompt and feedback.

**Input:**
```json
{
  "prompt": "The current prompt to improve",
  "feedback": {
    "score": 0.53,
    "weaknesses": ["Missing output format", "No error handling"],
    "suggestions": ["Add format spec", "Define edge cases"],
    "strengths": ["Clear role definition"]
  },
  "num_candidates": 3,
  "cross_pollination_prompts": ["optional other good prompts to learn from"]
}
```

**Output:**
```json
{
  "candidates": [
    {
      "prompt": "Improved prompt variant 1...",
      "strategy": "Added explicit output format specification",
      "addressed_weaknesses": ["Missing output format"]
    },
    {
      "prompt": "Improved prompt variant 2...",
      "strategy": "Added error handling and edge cases",
      "addressed_weaknesses": ["No error handling"]
    }
  ]
}
```

## Improvement Strategies

The optimizer applies various strategies:

1. **Add missing components** - Role, task, format, constraints, examples
2. **Fix specific weaknesses** - Directly address evaluation feedback
3. **Strengthen clarity** - Remove ambiguity, add specificity
4. **Add robustness** - Error handling, edge cases, constraints
5. **Cross-pollinate** - Borrow successful patterns from other prompts

## Example Usage

**Input prompt**: "You are a helpful coding assistant."

**Feedback**:
- Weaknesses: Missing format, no constraints, too vague
- Suggestions: Add output format, define supported languages

**Generated candidates**:

```
Candidate 1 (Format-focused):
"You are a Python coding assistant. 

When given a coding task:
1. Write clean, working Python code
2. Use markdown code blocks for output
3. Include brief comments for complex logic

If the request is unclear, ask for clarification."

Candidate 2 (Robustness-focused):
"You are an expert coding assistant specializing in Python.

## Input
Natural language description of a coding task.

## Output
Working Python code in markdown blocks.

## Constraints
- Only Python 3.11+ syntax
- No external dependencies unless specified
- If task is impossible, explain why"
```

## Cross-Pollination

When provided with successful prompts from other optimizers:

```json
{
  "cross_pollination_prompts": [
    "You are X. Given Y, output Z. Format: ..."
  ]
}
```

The optimizer will:
1. Identify successful patterns (structure, phrasing, components)
2. Adapt relevant patterns to the current prompt
3. Create hybrid candidates combining multiple approaches

## Model Configuration

Uses LiteLLM. Supports:
- `AZURE_OPENAI_*` environment variables
- `OLLAMA_API_BASE` for local models
