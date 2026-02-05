"""Basic prompt evaluator."""

import os
from dotenv import load_dotenv

load_dotenv()


def get_default_model() -> str:
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return f"azure/{os.getenv('AZURE_OPENAI_MODEL', 'gpt-4o-mini')}"
    return "gpt-4o-mini"


async def evaluate_prompt(prompt: str, examples: list, model: str = None) -> dict:
    """Evaluate a prompt against examples."""
    model = model or get_default_model()
    # Basic stub - returns placeholder
    return {
        "score": 0.5,
        "passed": False,
        "feedback": "Evaluation pending implementation",
        "strengths": [],
        "weaknesses": ["Not yet implemented"],
    }
