# Promptune — Prompt Optimization Skill

You have access to Promptune, a prompt optimization system where YOU act as both **tuner** and **judge**. Use it whenever the user asks to optimize, tune, evaluate, or improve an LLM system prompt.

## MCP Resources

### Tool

| Tool | Purpose |
|------|---------|
| `run_prompt_tests` | Run a prompt against training examples on the target model. Returns raw `{input, expected_output, actual_output}` for you to judge. |

### Prompts

| Prompt | Purpose |
|--------|---------|
| `judge_results` | Score and analyze test results. Produces scoring report with strengths, weaknesses, suggestions. |
| `feedback_rewrite` | Rewrite a prompt based on scoring feedback. Primary optimizer. |
| `example_augmentation` | Inject few-shot examples and DO/DON'T sections from training data. |
| `adversarial_hardening` | Harden a prompt against edge cases and failure patterns. |
| `clarity_rewrite` | Fix ambiguous or vague instructions with precise specifications. |

## Workflow

### Step 1: Setup — `.env` + `promptune.yaml` (target model + batch_size)

### Step 2: Find the prompt in the codebase

### Step 3: Prepare training data → `datasets/<feature>.json`

### Step 4: Baseline — `run_prompt_tests` → `judge_results` → save report to markdown

### Step 5: Optimize in parallel
Apply **all chosen optimizers simultaneously** on the original prompt using MCP prompts:
- `feedback_rewrite` — with scoring report
- `example_augmentation` — with training examples
- `adversarial_hardening` — with scoring report + failure patterns
- `clarity_rewrite` — with scoring report

Each produces one independent candidate. Run **parallel sub-agents** per candidate if possible.

### Step 6: Test & score all candidates
`run_prompt_tests` for each → `judge_results` for each → note scores, strengths, weaknesses, failures.

### Step 7: Select & iterate
- Take top **2×n** candidates by score, randomly select **n** (diversity + quality).
- **Inject good/bad findings** into optimizer context for next round.
- Repeat Steps 5-7 until converged (score 85%+ or no improvement for 2-3 rounds).

### Step 8: Present best prompt + diff + score trajectory → update codebase once approved.

## Tips

- Start with negative examples — fastest path to improvement.
- Be harsh when judging — the `judge_results` prompt enforces this.
- All optimizers in parallel, not sequential.
- Inject findings into next round for compounding improvement.
- Read `skills/promptune/SKILL.md` for the full detailed workflow.

