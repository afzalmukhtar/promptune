---
name: beam_orchestrator
description: Coordinates prompt optimization through beam search. Maintains multiple candidate prompts, evaluates them, and uses optimizers to generate improvements. The core Promptune algorithm.
---

# Beam Orchestrator

## Purpose

Coordinate the full prompt optimization loop using beam search. This is the core Promptune algorithm that ties together the evaluator and optimizers.

## How It Works

1. **Initialize beam** with starting prompt(s)
2. **Evaluate** all candidates in beam
3. **Generate variations** using optimizers (meta-prompt, few-shot)
4. **Evaluate** new candidates
5. **Select top-k** to form new beam
6. **Repeat** until convergence or max iterations

## Key Concepts

### Beam Search
Maintains multiple promising candidates (beam width = k), exploring several paths simultaneously rather than committing to one.

### Cross-Pollination
Best ideas from one optimizer can inform another - the meta-prompt optimizer might learn patterns from few-shot selections.

### Convergence
Stop when:
- Score improvement < threshold for N iterations
- Max iterations reached
- Target score achieved

## When to Use

- **Full optimization**: When you want the best possible prompt
- **Exploration**: When you're unsure which approach will work
- **Automated tuning**: When you want hands-off optimization

## MCP Tools Available

### optimize

Run the full beam search optimization loop.

**Input:**
```json
{
  "initial_prompt": "Starting prompt to optimize",
  "training_examples": [
    {"input": "...", "expected_output": "..."}
  ],
  "config": {
    "beam_width": 3,
    "max_iterations": 5,
    "target_score": 0.85,
    "convergence_threshold": 0.02,
    "optimizers": ["meta_prompt", "few_shot"]
  }
}
```

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
  "training_examples": [...],
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
    
    for iteration in range(config.max_iterations):
        # Evaluate current beam
        scores = evaluate_all(beam, examples)
        
        # Check convergence
        if max(scores) >= config.target_score:
            return beam[argmax(scores)]
        
        # Generate candidates from each beam member
        candidates = []
        for prompt, score in zip(beam, scores):
            feedback = get_feedback(prompt, score)
            
            # Meta-prompt optimizer
            candidates += meta_prompt_optimize(prompt, feedback)
            
            # Few-shot optimizer (rebuild examples)
            candidates += few_shot_optimize(prompt, examples)
        
        # Cross-pollination: share best patterns
        if len(beam) > 1:
            best_prompt = beam[argmax(scores)]
            for candidate in candidates:
                candidate = cross_pollinate(candidate, best_prompt)
        
        # Evaluate all candidates
        all_candidates = beam + candidates
        all_scores = evaluate_all(all_candidates, examples)
        
        # Select top-k for new beam
        beam = select_top_k(all_candidates, all_scores, k=config.beam_width)
        
        # Check for convergence (no improvement)
        if improvement < config.convergence_threshold:
            convergence_count += 1
            if convergence_count >= 3:
                break
    
    return beam[0]  # Best prompt
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| beam_width | 3 | Number of candidates to keep per iteration |
| max_iterations | 10 | Maximum optimization rounds |
| target_score | 0.90 | Stop if this score is reached |
| convergence_threshold | 0.02 | Minimum improvement to continue |
| optimizers | ["meta_prompt", "few_shot"] | Which optimizers to use |

## Model Configuration

Uses LiteLLM. Inherits model config from child components (evaluator, optimizers).
