# Promptune — Prompt Optimization Skill

You have access to Promptune, an automated prompt optimization system exposed via MCP tools. Use it whenever the user asks to optimize, tune, evaluate, or improve an LLM system prompt.

## MCP Tools Available

| Tool | Purpose |
|------|---------|
| `evaluate` | Score a prompt against training examples. Returns score, strengths, weaknesses, suggestions. |
| `optimize` | Run full automated optimization loop. Returns best prompt, score, and iteration history. |
| `optimization_step` | Run a single optimization step for fine-grained control. |

## Workflow

### Step 0: Setup Environment

1. Ensure `.env` exists with the correct API key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).
2. Configure `promptune.yaml` with model choices:
   - **target**: Model the prompt runs on in production (cheap/fast)
   - **tuner**: Generates improved prompts (smart/creative)
   - **judge**: Scores outputs (accurate/consistent)
3. Choose optimizers based on task:
   - All 5 for thorough optimization
   - `meta_prompt` + `adversarial` for quick hardening
   - `meta_prompt` + `few_shot` for example-heavy tasks
   - `meta_prompt` + `clarity_rewriter` for ambiguous prompts

### Step 1: Find the Prompt

Locate the system prompt string in the codebase. Note the file, variable, and what the LLM feature does.

### Step 2: Prepare Training Data

Create a JSON dataset in `datasets/<feature>.json` with 5-20 examples.

**Positive examples** (input + expected output):
```json
[{"input": "Write hello world", "expected_output": "print('Hello, World!')"}]
```

**Negative examples** (input + bad output + reason):
```json
[{"input": "Summarize this", "bad_output": "I don't know", "reason_why_bad": "Refused to attempt"}]
```

**Mixed** (both in one file):
```json
{
  "examples": [{"input": "...", "expected_output": "..."}],
  "negative_examples": [{"input": "...", "bad_output": "...", "reason_why_bad": "..."}]
}
```

### Step 3: Evaluate Current Prompt (Optional)

Call `evaluate` with the current prompt and training examples to get a baseline score.

### Step 4: Run Optimization

Call `optimize` with the initial prompt, training examples, and optionally negative examples.

### Step 5: Review & Replace

Present the optimized prompt and score to the user. Highlight changes. Once approved, update the prompt in the codebase.

### Step 6: Iterate

If the score is insufficient, refine training data and re-run. Use `optimization_step` for fine-grained control.

## The 5 Optimizers

| Optimizer | What It Does |
|-----------|-------------|
| `meta_prompt` | Rewrites prompt based on evaluation feedback + prompt understanding |
| `few_shot` | Selects optimal examples from training data to embed in prompt |
| `adversarial` | Hardens prompt against edge cases using negative examples |
| `example_augmentor` | Injects DO/DON'T sections from positive + negative examples |
| `clarity_rewriter` | Fixes ambiguous instructions using prompt understanding analysis |

## Tips

- Start with negative examples if available — fastest path to improvement.
- Use `sample_width: 2, max_iterations: 3` for fast feedback, then increase for thorough optimization.
- Use `optimization_step` when you want to try specific optimizers or manually curate candidates between iterations.
- Read `skills/promptune/SKILL.md` for the full detailed workflow.
