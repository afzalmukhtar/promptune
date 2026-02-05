"""Few-shot example selector."""

import json
import os
from dotenv import load_dotenv
from litellm import acompletion

load_dotenv()


def get_default_model() -> str:
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return f"azure/{os.getenv('AZURE_OPENAI_MODEL', 'gpt-4o-mini')}"
    return "gpt-4o-mini"


async def select_examples(prompt: str, pool: list[dict], num: int = 3, model: str = None) -> dict:
    """Select optimal few-shot examples."""
    model = model or get_default_model()
    
    if len(pool) <= num:
        return {"selected": pool, "reasoning": "Used all available"}
    
    score_prompt = f"""Rank examples by relevance. Return JSON: {{"rankings": [indices]}}

PROMPT: {prompt}
EXAMPLES: {json.dumps(pool[:10], indent=2)}"""

    response = await acompletion(model=model, messages=[{"role": "user", "content": score_prompt}], temperature=0.0)
    
    try:
        indices = json.loads(response.choices[0].message.content).get("rankings", [])[:num]
        return {"selected": [pool[i] for i in indices if i < len(pool)], "reasoning": "LLM-ranked"}
    except:
        return {"selected": pool[:num], "reasoning": "Fallback"}
