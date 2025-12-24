---
name: promptune
description: Automated prompt optimization where YOU (the agent) act as both tuner and judge. Use the run_prompt_tests MCP tool to test prompts against the target model, then evaluate results and generate improved prompts yourself.
---

# Promptune — Prompt Optimization Skill

## Overview

Promptune optimizes LLM system prompts by testing them against training examples on the **target model**. You (the agent) drive the entire process:

- **You are the Judge** — You analyze the target model's outputs and decide how well they match expectations.
- **You are the Tuner** — You generate improved prompt candidates based on your analysis.
- **The MCP tool runs the target model** — `run_prompt_tests` is the only tool. It runs a prompt against examples on the target model and returns raw outputs for you to judge.

## MCP Tool

| Tool | Purpose |
|------|---------|
| `run_prompt_tests` | Run a prompt against training examples on the target model. Returns raw `{input, expected_output, actual_output}` for you to judge. |

## Agent Workflow

### Step 0: Setup Environment

1. Check that `.env` exists with the correct API key for the target model provider.
2. Configure `promptune.yaml` with ONLY the target model:

```yaml
models:
  target: "gpt-4o-mini"  # The model the prompt runs on in production

optimization:
  batch_size: 5  # Examples to sample per test run
```

### Step 1: Find the Prompt

Locate the system prompt string in the codebase. Note the file, variable, and what the LLM feature does.

### Step 2: Prepare Training Data

Create a JSON dataset in `datasets/<feature>.json` with 5-20 examples.

**Positive examples** (input + expected output):
```json
[
  {"input": "Write hello world in Python", "expected_output": "print('Hello, World!')"},
  {"input": "Reverse the string 'hello'", "expected_output": "'olleh'"}
]
```

**Negative examples** (input + bad output + reason):
```json
[
  {"input": "Summarize this article", "bad_output": "I don't know", "reason_why_bad": "Refused to attempt the task"},
  {"input": "Write a poem", "bad_output": "Here is a poem:\nRoses are red...", "reason_why_bad": "Too generic, no creativity"}
]
```

**Mixed** (both in one file):
```json
{
  "examples": [{"input": "...", "expected_output": "..."}],
  "negative_examples": [{"input": "...", "bad_output": "...", "reason_why_bad": "..."}]
}
```

### Step 3: Test the Current Prompt (Baseline)

Call `run_prompt_tests` to see how the current prompt performs:

```json
{
  "prompt": "You are a coding assistant.",
  "training_examples": [
    {"input": "Write hello world", "expected_output": "print('Hello, World!')"}
  ]
}
```

Returns raw results:
```json
{
  "positive_results": [
    {"input": "Write hello world", "expected_output": "print('Hello, World!')", "actual_output": "..."}
  ],
  "negative_results": []
}
```

### Step 4: Judge the Results (You Are the Judge)

Analyze each test result yourself using this rubric:

**For positive examples**, compare `actual_output` vs `expected_output`:
1. **Semantic match** — Does the actual output accomplish the same goal?
2. **Format match** — Is the format similar (code vs prose, structure)?
3. **Correctness** — Is the output factually/functionally correct?
4. **Completeness** — Does it fully address the task?

**For negative examples**, compare `actual_output` vs `bad_output`:
1. **Avoids bad pattern** — Does the output avoid the failure described in `reason_why_bad`?
2. **Different from bad** — Is it meaningfully different from the known bad output?
3. **Better quality** — Is it actually a good response, not just differently bad?

**Structural analysis** of the prompt itself:
- Does it define a **role** (who the AI is)?
- Does it specify the **task** (what to do)?
- Does it describe the **format** (how to structure output)?
- Does it set **constraints** (rules and boundaries)?
- Does it include **examples** (input/output pairs)?

Synthesize your analysis into:
- **Score** (0-100): Overall quality assessment
- **Strengths**: What the prompt does well
- **Weaknesses**: What needs fixing
- **Suggestions**: Specific, actionable improvements

### Step 5: Generate Improved Prompts (You Are the Tuner)

Based on your analysis, generate 1-3 improved prompt candidates. Apply these optimization strategies:

#### Strategy 1: Feedback-Driven Rewriting
- Fix every weakness you identified
- Implement every suggestion
- Keep all strengths intact
- Make the prompt more detailed and explicit

#### Strategy 2: Few-Shot Example Injection
- Select the most relevant training examples
- Embed them directly in the prompt as input/output demonstrations
- Balance relevance, diversity, and complexity

#### Strategy 3: Adversarial Hardening
- Identify edge cases the prompt doesn't handle
- Add explicit constraints for those cases
- Add "DO NOT" rules for known failure patterns
- If negative examples exist, add a "What BAD output looks like — AVOID" section

#### Strategy 4: DO/DON'T Augmentation
- Add a "What GOOD output looks like" section with positive examples
- Add a "What BAD output looks like — AVOID" section with negative examples and reasons
- Naturally integrate examples, don't just append

#### Strategy 5: Clarity Rewriting
- Find ambiguous or vague instructions
- Replace vague quantifiers ("some", "a few") with specifics
- Add missing details (format specs, edge cases, boundaries)

**Prompt structure checklist** — include all relevant sections:
```
[ROLE] - Who the AI is
[TASK] - What to do, step by step
[FORMAT] - Exact output format expected
[CONSTRAINTS] - Rules, limitations, boundaries
[EXAMPLES] - At least one input/output example
[ERROR HANDLING] - What to do when things go wrong
```

### Step 6: Test Improved Prompts

Call `run_prompt_tests` with each improved prompt candidate. Judge the results using the same rubric from Step 4.

### Step 7: Select Best & Iterate

Compare scores across all candidates (original + improved). Pick the best one.

**If satisfied**: Present the best prompt and score to the user. Highlight what changed. Once approved, update the codebase.

**If not satisfied**: Take the best candidate as your new starting point and repeat from Step 4. Common iteration strategies:
- Add more training examples targeting specific failure cases
- Try a different optimization strategy
- Combine strengths from multiple candidates
- Focus on the weakest area (structural, empirical, etc.)

**Convergence heuristics** — stop iterating when:
- Score is above 85-90%
- No meaningful improvement for 2-3 rounds
- The user is satisfied with the results

## Tips

- **Start with negatives**: If you have examples of bad outputs, they're the fastest path to improvement.
- **Be harsh when judging**: Generous scoring leads to weak optimization. Be critical.
- **One strategy at a time**: Don't try all 5 strategies at once. Pick 1-2, test, then try others.
- **Keep what works**: Never remove parts of the prompt that are working well.
- **Test on the actual target model**: The target model in config should be the exact model used in production.
