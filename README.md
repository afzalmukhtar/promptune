# Promptune

**Prompt optimization where your AI agent is both tuner and judge — powered by MCP.**

Promptune is a skill + MCP tool that your AI coding agent uses to optimize system prompts. The agent reads the skill, prepares training data, tests prompts against the target model, judges the results, and generates improved prompts — all by itself.

## How It Works

1. **Agent reads** `skills/promptune/SKILL.md` to understand the workflow
2. **Agent prepares** training data (positive and/or negative examples)
3. **Agent calls** `run_prompt_tests` to test prompts against the target model
4. **Agent judges** the raw outputs (compares actual vs expected)
5. **Agent generates** improved prompts based on its analysis
6. **Agent iterates** until the prompt is optimized

The agent is both the **tuner** (generates improved prompts) and the **judge** (evaluates outputs). The MCP server only runs the target model.

## MCP Tool

| Tool | Purpose |
|------|---------|
| **`run_prompt_tests`** | Run a prompt against training examples on the target model. Returns raw `{input, expected_output, actual_output}` for the agent to judge. |

## Setup

### 1. Create `promptune.yaml`

```yaml
models:
  target: "gpt-4o-mini"  # The model the prompt runs on in production

optimization:
  batch_size: 5  # Examples to sample per test run
```

### 2. Create `.env` with API keys

```bash
OPENAI_API_KEY=your-key
# Or: ANTHROPIC_API_KEY=your-key
# Or: OLLAMA_API_BASE=http://localhost:11434
```

### 3. Register the MCP server

Point your agent's MCP config at:
```
mcp_servers/orchestrator/server.py
```

## Training Data

### Positive examples
```json
[
  {"input": "Write hello world", "expected_output": "print('Hello, World!')"},
  {"input": "Add two numbers", "expected_output": "def add(a, b): return a + b"}
]
```

### Negative examples
```json
[
  {"input": "Write a function", "bad_output": "I can't do that", "reason_why_bad": "Refusal to attempt"},
  {"input": "Summarize this", "bad_output": "ok", "reason_why_bad": "Too short, no actual summary"}
]
```

### Mixed (both in one file)
```json
{
  "examples": [{"input": "...", "expected_output": "..."}],
  "negative_examples": [{"input": "...", "bad_output": "...", "reason_why_bad": "..."}]
}
```

Save to `datasets/<feature>.json`. Aim for 5-20 examples.

## Included Datasets

| Dataset | Examples | Negatives | Use Case |
|---------|----------|-----------|----------|
| `code_assistant.json` | 51 | 0 | Python code generation |
| `text_summarizer.json` | 51 | 0 | News/text summarization |
| `sentiment_classifier.json` | 52 | 0 | Sentiment analysis |
| `email_writer_with_negatives.json` | 25 | 21 | Professional email writing |
| `math_tutor.json` | 52 | 0 | Math problem solving |

## Agent Optimization Strategies

The agent applies these strategies when generating improved prompts:

| Strategy | What It Does |
|----------|-------------|
| **Feedback-driven rewriting** | Fix weaknesses, keep strengths, make prompt more detailed |
| **Few-shot injection** | Embed relevant training examples directly in the prompt |
| **Adversarial hardening** | Add constraints for edge cases and failure patterns |
| **DO/DON'T augmentation** | Add good/bad output sections from training data |
| **Clarity rewriting** | Fix ambiguous instructions, add missing specifics |

## Architecture

```
promptune/
├── promptune.yaml               # Config: target model + batch size
├── .env                         # API keys
├── datasets/                    # Training datasets (JSON)
├── skills/promptune/SKILL.md    # Agent skill doc (start here)
├── mcp_servers/
│   ├── orchestrator/            # MCP server (run_prompt_tests)
│   ├── evaluator/               # Target model test runner
│   ├── targets/                 # Custom evaluation targets
│   └── utils/                   # Config, LLM calls, logging
└── schemas/                     # Shared Pydantic models
```

## Installation

```bash
pip install -e ".[dev]"
```

## License

MIT
