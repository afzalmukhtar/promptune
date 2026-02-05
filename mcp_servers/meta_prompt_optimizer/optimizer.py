"""Meta-prompt optimizer - LLM-based improvement."""

import json
import os
from dotenv import load_dotenv
from litellm import acompletion

load_dotenv()


def get_default_model() -> str:
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return f"azure/{os.getenv('AZURE_OPENAI_MODEL', 'gpt-4o-mini')}"
    return "gpt-4o-mini"


async def optimize(prompt: str, feedback: dict, num_candidates: int = 2, model: str = None) -> dict:
    """Generate improved prompt candidates."""
    model = model or get_default_model()
    
    opt_prompt = f"""Improve this prompt based on feedback.

ORIGINAL: {prompt}
SCORE: {feedback.get('score', 0)}
WEAKNESSES: {feedback.get('weaknesses', [])}
SUGGESTIONS: {feedback.get('suggestions', [])}

Generate {num_candidates} improved versions. Return JSON: {{"candidates": [{{"prompt": "...", "strategy": "..."}}]}}"""

    response = await acompletion(
        model=model,
        messages=[{"role": "user", "content": opt_prompt}],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"candidates": []}
