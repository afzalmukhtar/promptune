---
name: promptune
description: Automated prompt optimization where YOU (the agent) act as both tuner and judge. Use the run_prompt_tests tool to test prompts and MCP prompts to guide your judging and optimization.
---

# Promptune — Prompt Optimization Skill

## Overview

Promptune optimizes LLM system prompts by testing them against training examples on the **target model**. You (the agent) drive the entire process:

- **You are the Judge** — Use the `judge_results` prompt to analyze outputs and score them.
- **You are the Tuner** — Use the 4 optimization prompts to generate improved candidates **in parallel**.
- **The MCP tool runs the target model** — `run_prompt_tests` returns raw outputs for you to judge.

## MCP Resources

### Tool

| Tool | Purpose |
|------|---------|
| `run_prompt_tests` | Run a prompt against training examples on the target model. Returns raw `{input, expected_output, actual_output}`. |

### Prompts

| Prompt | Purpose |
|--------|---------|
| `judge_results` | Score and analyze test results. Produces a scoring report with strengths, weaknesses, suggestions. |
| `feedback_rewrite` | Rewrite a prompt based on scoring feedback. Fixes weaknesses, preserves strengths. |
| `example_augmentation` | Inject few-shot examples and DO/DON'T sections from training data. |
| `adversarial_hardening` | Harden a prompt against edge cases, failure patterns, and adversarial inputs. |
| `clarity_rewrite` | Fix ambiguous, vague, or unclear instructions with precise specifications. |

## Agent Workflow

### Step 1: Setup Environment

1. Check that `.env` exists with the correct API key for the target model provider.
2. Configure `promptune.yaml`:

```yaml
models:
  target: "gpt-4o-mini"  # The model the prompt runs on in production

optimization:
  batch_size: 5  # Examples to sample per test run
```

### Step 2: Find the Prompt

Locate the system prompt string in the codebase. Note the file, variable, and what the LLM feature does.

### Step 3: Prepare Training Data

Create a JSON dataset in `datasets/<feature>.json` with 5-20 examples and save it.

**Positive**: `[{"input": "...", "expected_output": "..."}]`
**Negative**: `[{"input": "...", "bad_output": "...", "reason_why_bad": "..."}]`
**Mixed**: `{"examples": [...], "negative_examples": [...]}`

### Step 4: Run Baseline & Score (Initial Scoring)

1. Call `run_prompt_tests` with the original prompt and training data.
2. Use the `judge_results` prompt to score the results.
3. **Save the original prompt + scoring report** to a markdown file (e.g., `optimization_log.md`) for tracking.

### Step 5: Optimize in Parallel

Apply **all chosen optimizers simultaneously** on the **original prompt** (not sequentially):

1. Choose which optimizers to apply (1 or more of the 4 strategies).
2. Run each optimizer **in parallel** using the corresponding MCP prompt:
   - `feedback_rewrite` — with the scoring report from Step 4
   - `example_augmentation` — with training examples
   - `adversarial_hardening` — with scoring report + failure patterns from judging
   - `clarity_rewrite` — with scoring report
3. Each optimizer produces one candidate prompt independently from the original.

**If possible, run parallel sub-agents for each candidate** to speed up the process.

### Step 6: Test & Score All Candidates

1. Call `run_prompt_tests` for **each** candidate prompt (in parallel if possible).
2. Use `judge_results` to score each candidate's results.
3. For each candidate, note:
   - **Score** (0-100)
   - **Strengths** (what improved vs original)
   - **Weaknesses** (what's still broken)
   - **Failure patterns** (specific inputs that failed)

### Step 7: Select Candidates & Iterate

**Candidate selection** (top-2n random sampling):
1. Let `n` = number of candidates you want to keep (typically 2-3).
2. Take the **top 2×n** candidates by score.
3. **Randomly select n** from those top 2×n (adds diversity, prevents local optima).

**Next iteration:**
1. For each selected candidate, **inject the good and bad findings** from Step 6 into the optimization context.
2. Run **all chosen optimizers again in parallel** on each candidate — this time using the enriched feedback.
3. Test and score the new candidates (Step 6).
4. Repeat selection (Step 7).

**Convergence** — stop iterating when:
- Score is above 85-90%
- No meaningful improvement for 2-3 rounds
- The user is satisfied

### Step 8: Review & Replace

1. Present the **best prompt** and its score to the user.
2. Show a **diff** vs the original prompt — what changed and why.
3. Show the **score improvement** trajectory.
4. Once approved, update the prompt in the codebase.

## Optimization Strategies (The 4 Optimizers)

### 1. Feedback-Driven Rewriting (`feedback_rewrite`)
The primary optimizer. Takes the scoring report and rewrites the prompt to fix every weakness while preserving strengths. Best for: general improvement, fixing specific failures.

### 2. Example Augmentation (`example_augmentation`)
Injects few-shot examples and DO/DON'T sections from training data. Selects diverse, representative examples. Best for: tasks where the model needs concrete demonstrations.

### 3. Adversarial Hardening (`adversarial_hardening`)
Hardens the prompt against edge cases and failure patterns. Adds guardrails, constraints, and explicit error handling. Best for: production prompts that must handle diverse inputs.

### 4. Clarity Rewriting (`clarity_rewrite`)
Finds and fixes ambiguous, vague, or unclear instructions. Replaces fuzzy language with precise specifications. Best for: complex prompts with many instructions.

## Tips

- **Start with negatives**: If you have examples of bad outputs, they're the fastest path to improvement.
- **Be harsh when judging**: Generous scoring leads to weak optimization. Use the `judge_results` prompt — it enforces harsh scoring.
- **All optimizers in parallel**: Run all chosen optimizers on the same prompt simultaneously, not sequentially. Each produces an independent candidate.
- **Inject findings into next round**: Feed strengths and weaknesses from scoring back into the optimizer prompts for the next iteration.
- **Use sub-agents**: If the IDE supports parallel sub-agents, spawn one per candidate for concurrent optimization.
- **Save progress**: Keep a running optimization log in markdown so you can track score trajectory across iterations.
