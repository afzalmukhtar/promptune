---
name: few_shot_optimizer
description: Optimizes prompts by selecting and ordering the best few-shot examples. Uses semantic similarity and diversity to pick examples that maximize prompt effectiveness.
---

# Few-Shot Optimizer

## Purpose

Improve prompt performance by intelligently selecting and ordering few-shot examples. Not all examples are equal - the right examples in the right order can dramatically improve results.

## How It Works

1. Takes a pool of training examples
2. Scores examples by relevance (semantic similarity to task)
3. Scores examples by diversity (cover different cases)
4. Selects optimal subset that balances relevance + diversity
5. Orders examples for maximum learning (simple → complex)

## When to Use

- **Example selection**: When you have many examples but can only use a few
- **Improving consistency**: When outputs vary too much
- **Balancing coverage**: When you need examples covering different cases
- **Prompt length optimization**: When you need to fit in token limits

## MCP Tools Available

### select_examples

Select the optimal few-shot examples from a pool.

**Input:**
```json
{
  "prompt": "The base prompt (without examples)",
  "example_pool": [
    {"input": "example 1 input", "expected_output": "example 1 output"},
    {"input": "example 2 input", "expected_output": "example 2 output"}
  ],
  "num_examples": 3,
  "strategy": "balanced"
}
```

**Strategies:**
- `"balanced"` - Balance relevance and diversity (default)
- `"relevant"` - Prioritize most relevant examples
- `"diverse"` - Prioritize covering different cases
- `"simple_first"` - Order from simple to complex

**Output:**
```json
{
  "selected_examples": [
    {"input": "...", "expected_output": "...", "score": 0.92},
    {"input": "...", "expected_output": "...", "score": 0.87}
  ],
  "prompt_with_examples": "Full prompt with examples formatted and inserted",
  "selection_reasoning": "Why these examples were chosen"
}
```

### format_examples

Format selected examples into a prompt section.

**Input:**
```json
{
  "examples": [{"input": "...", "expected_output": "..."}],
  "format_style": "markdown"
}
```

**Format styles:**
- `"markdown"` - Markdown headers and code blocks
- `"xml"` - XML tags
- `"numbered"` - Numbered list
- `"chat"` - User/Assistant format

## Selection Algorithm

### Step 1: Score Relevance
Each example is scored by how well it represents the task:
- Semantic similarity to prompt description
- Coverage of key concepts mentioned in prompt

### Step 2: Score Diversity  
Examples are scored by how different they are from each other:
- Input diversity (different types of inputs)
- Output diversity (different output patterns)
- Complexity diversity (easy, medium, hard)

### Step 3: Select Optimal Set
Greedy selection balancing relevance and diversity:
```
for i in range(num_examples):
    best = argmax(relevance[e] + diversity_bonus[e] for e in remaining)
    selected.append(best)
    update_diversity_scores()  # Penalize similar examples
```

### Step 4: Order Examples
Sort selected examples by complexity (simple first):
- Shorter inputs/outputs first
- Less edge cases first
- More common patterns first

## Example Usage

**Pool of 10 examples**, need to select 3:

```
Pool:
1. "reverse 'hello'" → "'olleh'"  (simple, string)
2. "sort [3,1,2]" → "[1,2,3]"  (simple, list)
3. "reverse ''" → "''"  (edge case, empty)
4. "sort []" → "[]"  (edge case, empty)
5. "reverse 'a'" → "'a'"  (edge case, single)
6. "sort [1]" → "[1]"  (edge case, single)
7. "reverse 'hello world'" → "'dlrow olleh'"  (medium, space)
8. "sort [3,1,2,1]" → "[1,1,2,3]"  (medium, duplicates)
9. "reverse unicode" → complex  (hard)
10. "sort mixed types" → complex  (hard)

Selected (balanced):
1. "reverse 'hello'" - simple, common case
2. "sort [3,1,2]" - different operation, common case  
3. "reverse ''" - edge case coverage
```

## Model Configuration

Uses LiteLLM for embeddings and scoring:
- `AZURE_OPENAI_*` for Azure
- `OLLAMA_API_BASE` for local
