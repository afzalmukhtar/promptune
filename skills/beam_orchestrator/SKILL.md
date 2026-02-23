---
name: beam_orchestrator
description: Coordinates prompt optimization through beam search. Maintains multiple candidate prompts, evaluates them with 3 model roles, and uses 5 optimizers to generate improvements. Supports negative examples and prompt understanding feedback. The core Promptune algorithm.
---

# Beam Orchestrator

## Purpose

Coordinate the full prompt optimization loop using beam search. This is the core Promptune algorithm that ties together the evaluator and all 5 optimizers.

## How It Works

1. **Load config** from `promptune.yaml` (3 model roles + optimization params)
2. **Initialize beam** with starting prompt(s)
3. **Evaluate** all candidates (target model runs prompt, judge model scores)
4. **Generate variations** using enabled optimizers (tuner model)
5. **Evaluate** new candidates
6. **Select top-k** to form new beam
7. **Repeat** until convergence or max iterations

## Key Concepts

### 3 Model Roles
- **target** — The model the prompt runs on (cheap/fast)
- **tuner** — Generates improved prompts (smart/creative)
- **judge** — Scores outputs and analyzes understanding (accurate)

### 5 Optimizers
| Optimizer | What It Does |
|-----------|-------------|
| **meta_prompt** | Rewrites based on feedback + prompt understanding |
| **few_shot** | Selects optimal examples from training data |
| **adversarial** | Hardens against edge cases using negative examples |
| **example_augmentor** | Injects DO/DON'T sections with positive + negative examples |
| **clarity_rewriter** | Fixes ambiguous instructions using prompt understanding |

### Negative Examples & Evaluation Modes

Training data can include "bad output + reason why bad" pairs (`NegativeTrainingExample`). Three evaluation modes are automatically selected:

| Mode | Condition | Empirical Scoring |
|------|-----------|-------------------|
| **Standard** | Only positive examples | Test output vs expected output (match = good) |
| **Negative-only** | Only negative examples | Test output vs bad output (match = bad, inverted score) |
| **Mixed** | Both positive + negative | Average of standard + reverse empirical scores |

Negative examples are also passed to the **adversarial** and **example_augmentor** optimizers for candidate generation in all modes.

### Prompt Understanding
The evaluator analyzes which prompt sections the LLM followed vs ignored. This feedback is passed to meta_prompt, adversarial, and clarity_rewriter optimizers.

### Convergence
Stop when:
- Score improvement < threshold for N iterations
- Max iterations reached
- Target score achieved

## MCP Tools Available

### optimize

Run the full beam search optimization loop. Supports positive-only, negative-only, or mixed examples.

**Input:**
```json
{
  "initial_prompt": "Starting prompt to optimize",
  "training_examples": [
    {"input": "...", "expected_output": "..."}
  ],
  "negative_examples": [
    {"input": "...", "bad_output": "...", "reason_why_bad": "..."}
  ],
  "config_path": "promptune.yaml"
}
```

- Pass only `training_examples` → standard empirical scoring
- Pass only `negative_examples` → reverse empirical scoring (avoids bad patterns)
- Pass both → combined scoring (average of positive + negative)

**Output:**
```json
{
  "best_prompt": "The optimized prompt...",
  "best_score": 0.87,
  "iterations": 4,
  "converged": true,
  "history": [
    {"iteration": 1, "best_score": 0.53, "beam_scores": [0.53, 0.48, 0.45]},
    {"iteration": 2, "best_score": 0.71, "beam_scores": [0.71, 0.68, 0.65]}
  ]
}
```

### step

Run a single optimization step (for manual control).

**Input:**
```json
{
  "beam": ["prompt1", "prompt2", "prompt3"],
  "training_examples": [{"input": "...", "expected_output": "..."}],
  "negative_examples": [{"input": "...", "bad_output": "...", "reason_why_bad": "..."}],
  "config_path": "promptune.yaml",
  "optimizers": ["meta_prompt"]
}
```

**Output:**
```json
{
  "new_beam": ["improved1", "improved2", "improved3"],
  "scores": [0.75, 0.72, 0.70],
  "candidates_generated": 6,
  "candidates_evaluated": 6
}
```

## Algorithm Detail

```
function beam_optimize(initial_prompt, examples, config):
    beam = [initial_prompt]
    
    for iteration in range(config.optimization.max_iterations):
        # Evaluate current beam (target model runs, judge model scores)
        # Random batch of batch_size examples per iteration
        scores, understandings = evaluate_all(beam, examples, iteration)
        
        # Check convergence
        if max(scores) >= config.optimization.target_score:
            return beam[argmax(scores)]
        
        # Generate candidates from each beam member
        candidates = []
        for prompt, score, understanding in zip(beam, scores, understandings):
            feedback = get_feedback(prompt, score)
            
            # Each enabled optimizer generates candidates
            if "meta_prompt" in optimizers:
                candidates += meta_prompt(prompt, feedback, understanding)
            if "few_shot" in optimizers:
                candidates += few_shot(prompt, examples)
            if "adversarial" in optimizers:
                candidates += adversarial(prompt, feedback, negatives, understanding)
            if "example_augmentor" in optimizers:
                candidates += augment(prompt, positives, negatives)
            if "clarity_rewriter" in optimizers:
                candidates += clarity_rewrite(prompt, understanding)
        
        # Cross-pollination: share best patterns
        if len(beam) > 1:
            best_prompt = beam[argmax(scores)]
            for candidate in candidates:
                candidate = cross_pollinate(candidate, best_prompt)
        
        # Evaluate all candidates
        all_candidates = beam + candidates
        all_scores = evaluate_all(all_candidates, examples, iteration)
        
        # Select top-k for new beam
        beam = select_top_k(all_candidates, all_scores, k=config.optimization.beam_width)
    
    return beam[0]  # Best prompt
```

## Configuration

All settings come from `promptune.yaml`:

```yaml
models:
  target: "azure/gpt-4o-mini"
  tuner: "azure/gpt-4o"
  judge: "azure/gpt-4o-mini"

optimization:
  beam_width: 3
  max_iterations: 10
  target_score: 0.90
  convergence_threshold: 0.02
  convergence_patience: 3
  batch_size: 5
  optimizers:
    - meta_prompt
    - few_shot
    - adversarial
    - example_augmentor
    - clarity_rewriter
```

| Option | Default | Description |
|--------|---------|-------------|
| beam_width | 3 | Number of candidates to keep per iteration |
| max_iterations | 10 | Maximum optimization rounds |
| target_score | 0.90 | Stop if this score is reached |
| convergence_threshold | 0.02 | Minimum improvement to continue |
| convergence_patience | 3 | Rounds without improvement before stopping |
| batch_size | 5 | Training examples per evaluation batch |
| optimizers | all 5 | Which optimizers to enable |
