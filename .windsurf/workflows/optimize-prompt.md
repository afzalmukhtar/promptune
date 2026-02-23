---
description: Optimize an LLM system prompt using Promptune's beam search with negative/positive examples
---

# Optimize a Prompt with Promptune

See `skills/evaluator/SKILL.md` for scoring details and `skills/beam_orchestrator/SKILL.md` for the optimization algorithm.

## Steps

1. **Find the prompt.** Locate the system prompt string in the codebase. Note the file, variable, and what the LLM feature does.

2. **Collect examples into `datasets/<feature>.json`.** Ask the user which they have:

   - **Negative-only** (easiest — just collect bad outputs):
     ```json
     [{"input": "...", "bad_output": "...", "reason_why_bad": "..."}]
     ```
   - **Positive-only** (ideal input/output pairs):
     ```json
     [{"input": "...", "expected_output": "..."}]
     ```
   - **Mixed** (best results — both in the same file):
     ```json
     [
       {"input": "...", "expected_output": "..."},
       {"input": "...", "bad_output": "...", "reason_why_bad": "..."}
     ]
     ```
   Aim for 5-20 examples.

3. **Run Promptune:**
// turbo
```bash
python test_promptune.py \
  --data datasets/<feature>.json \
  --prompt "<current system prompt>" \
  --beam 3 --max-iter 3 --batch-size 5 \
  --optimizers meta_prompt adversarial example_augmentor
```
   Quick: `--beam 2 --max-iter 2 --batch-size 3`. Thorough: `--beam 3 --max-iter 5 --batch-size 5`. Add `--save` to version outputs.

4. **Review.** Present the optimized prompt to the user. Highlight what changed vs the original.

5. **Replace in code.** Update the system prompt variable with the optimized version once approved.
