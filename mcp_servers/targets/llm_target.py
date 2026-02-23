"""Default LLM Target."""

import os

from dotenv import load_dotenv
from litellm import acompletion

from .base import BaseTarget

load_dotenv()


class LLMTarget(BaseTarget):
    """LLM target - prompt as system, input as user."""
    
    def __init__(self, model: str = None):
        self.model = model or os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")
    
    async def invoke(self, prompt: str, input_text: str) -> str:
        response = await acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
