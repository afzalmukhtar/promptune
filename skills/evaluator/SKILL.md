---
name: evaluator
description: Empirical prompt evaluator that actually TESTS prompts against examples. Combines real output testing (50%), structural analysis (30%), and adversarial critique (20%). Most trustworthy evaluation approach.
---

# Empirical Prompt Evaluator

## Purpose

Evaluate prompt quality by **actually testing it** against training examples. Don't guess if a prompt will work - run it and see.

## Key Design Principles

1. **Empirical testing** - Actually run the prompt, compare output to expected
2. **Parallel execution** - All analyses run concurrently for speed
3. **Weighted scoring** - Empirical results matter most (50%)
4. **Code-computed scores** - All math done in code, not by LLM

## When to Use

- **Reliable evaluation**: Ground truth from actual outputs
- **Comparing prompts**: Fair comparison based on real performance
- **Optimization feedback**: Know exactly what's working and what isn't
- **Pre-deployment validation**: Verify prompt works before shipping

## MCP Tools Available

### evaluate

Evaluate a prompt by testing it against examples.

**Input:**
```json
{
  "prompt": "The prompt text to evaluate",
  "training_examples": [
    {"input": "example input", "expected_output": "example output"}
  ],
  "model": "azure/gpt-4o-mini"
}
```

**Output:**
```json
{
  "prompt": "The evaluated prompt",
  "score": 0.53,
  "passed": false,
  "feedback": "Score: 53/100\n\nEmpirical (50%): 75/100 - Tested against 1 examples\nStructural (30%): 20/100 - 1/5 checks passed\nAdversarial (20%): 50/100 - 2 weaknesses found\n\nExample Results:\n  ✓ Example 1: 75/100\n\nAssessment: ...",
  "strengths": ["Has role"],
  "weaknesses": ["Missing: has format", "Missing: has constraints"],
  "suggestions": ["Add output format specification", "Define error handling"]
}
```

## Scoring System

### Weight Distribution
| Component | Weight | What it measures |
|-----------|--------|------------------|
| **Empirical** | 50% | Does the prompt actually produce correct outputs? |
| **Structural** | 30% | Does the prompt have required components? |
| **Adversarial** | 20% | Are there hidden weaknesses? |

### Empirical Testing (50%)

For each training example:
1. Run the prompt with the example input
2. Compare actual output to expected output
3. Score on 4 binary criteria (25 pts each):

| Check | Points | Question |
|-------|--------|----------|
| semantic_match | 25 | Same meaning/goal achieved? |
| format_match | 25 | Similar format/structure? |
| correctness | 25 | Factually/functionally correct? |
| completeness | 25 | Fully addresses the task? |

**Example score**: 75/100 = 3 of 4 checks passed

### Structural Analysis (30%)

5 binary checks, 20 points each:

| Check | Points | Question |
|-------|--------|----------|
| has_role | 20 | Defines WHO the AI should be? |
| has_task | 20 | Explains WHAT to do? |
| has_format | 20 | Specifies HOW to format output? |
| has_constraints | 20 | Sets boundaries or rules? |
| has_examples | 20 | Includes examples? |

### Adversarial Critique (20%)

Score based on weaknesses found:
- 0 weaknesses = 100 pts
- 1 weakness = 75 pts
- 2 weaknesses = 50 pts
- 3+ weaknesses = 25 pts

## Final Score Calculation

```python
final_score = (
    empirical_score * 0.50 +
    structural_score * 0.30 +
    adversarial_score * 0.20
)
```

**Pass threshold**: 70/100

## Example Evaluation

**Prompt**: "You are a helpful coding assistant."

**Results**:
```
Score: 53/100 (FAILED)

Empirical (50%): 75/100 - Tested against 1 example
  ✓ Example 1: 75/100
    - semantic_match: ✓
    - format_match: ✓  
    - correctness: ✓
    - completeness: ✗

Structural (30%): 20/100 - 1/5 checks passed
  ✓ has_role
  ✗ has_task
  ✗ has_format
  ✗ has_constraints
  ✗ has_examples

Adversarial (20%): 50/100 - 2 weaknesses found
  - "Extremely vague, lacks detail about language or context"
  - "No error handling or edge case guidance"

Assessment: This prompt is so generic it's practically useless 
for consistent coding assistance.
```

**Key insight**: The prompt *works* for simple cases (75% empirical) but lacks structure for reliability.

## Prompt Understanding Analysis

The evaluator now produces a **PromptUnderstanding** analysis showing which prompt sections the target LLM followed vs ignored:

```json
{
  "well_followed": [
    {"section": "Role definition", "evidence": "Acted as coding assistant", "score": 0.9}
  ],
  "poorly_followed": [
    {"section": "Output format", "evidence": "Did not use markdown", "score": 0.3, "reason": "Format spec was ambiguous"}
  ],
  "overall_compliance": 0.6
}
```

This analysis is passed to the **meta_prompt**, **adversarial**, and **clarity_rewriter** optimizers to target improvements.

## Configuration

Uses the **target** model (runs the prompt) and **judge** model (scores outputs) from `promptune.yaml`:

```yaml
models:
  target: "azure/gpt-4o-mini"   # Runs the prompt against inputs
  judge: "azure/gpt-4o-mini"    # Scores outputs, analyzes understanding
```

Random batch sampling is controlled by `batch_size`:

```yaml
optimization:
  batch_size: 5   # Examples evaluated per iteration
```
