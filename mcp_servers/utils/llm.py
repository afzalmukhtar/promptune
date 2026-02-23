"""
Structured LLM Call Helper.

Provides `call_llm_structured()` which uses tool/function calling to force
LLMs to return data matching a Pydantic schema. Falls back to JSON mode
if tool calling is not supported.

All judge and tuner LLM calls should use this helper.
Target model calls should use plain `litellm.acompletion()` for natural responses.
"""

import json
import logging
from typing import TypeVar

import litellm
from litellm import acompletion
from pydantic import BaseModel, ValidationError

# Drop unsupported params (e.g. temperature for GPT-5) instead of raising errors
litellm.drop_params = True

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _pydantic_to_tool_schema(model: type[BaseModel]) -> dict:
    """Convert a Pydantic model to an OpenAI-compatible tool schema.

    Args:
        model: A Pydantic BaseModel class.

    Returns:
        Tool definition dict for litellm.acompletion(tools=[...]).
    """
    json_schema = model.model_json_schema()

    # Remove pydantic-specific keys that OpenAI doesn't understand
    def _clean_schema(schema: dict) -> dict:
        cleaned = {}
        for key, value in schema.items():
            if key in ("title", "$defs"):
                continue
            if isinstance(value, dict):
                cleaned[key] = _clean_schema(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    _clean_schema(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                cleaned[key] = value
        return cleaned

    # Resolve $ref references inline
    defs = json_schema.get("$defs", {})

    def _resolve_refs(schema: dict) -> dict:
        if "$ref" in schema:
            ref_name = schema["$ref"].split("/")[-1]
            if ref_name in defs:
                return _resolve_refs(_clean_schema(defs[ref_name]))
            return schema

        result = {}
        for key, value in schema.items():
            if key == "$defs":
                continue
            if isinstance(value, dict):
                result[key] = _resolve_refs(value)
            elif isinstance(value, list):
                result[key] = [
                    _resolve_refs(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                result[key] = value
        return result

    resolved = _resolve_refs(json_schema)
    cleaned = _clean_schema(resolved)

    # Build tool name from model class name (snake_case)
    name = model.__name__
    tool_name = "".join(f"_{c.lower()}" if c.isupper() else c for c in name).lstrip("_")
    if len(tool_name) > 64:
        tool_name = tool_name[:64]

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": model.__doc__ or f"Return structured {name} data",
            "parameters": cleaned,
        },
    }


async def call_llm_structured(
    model: str,
    messages: list[dict],
    response_model: type[T],
    temperature: float = 0.0,
    max_retries: int = 2,
) -> T:
    """Call an LLM and force structured output matching a Pydantic model.

    Uses tool/function calling with forced tool_choice to guarantee the LLM
    returns data matching the schema. Falls back to JSON mode on tool call failure.

    Args:
        model: LiteLLM model string (e.g. 'azure/gpt-4o-mini').
        messages: Chat messages to send.
        response_model: Pydantic BaseModel class defining expected output schema.
        temperature: LLM temperature (default 0.0 for deterministic).
        max_retries: Number of retries on parse/validation failure.

    Returns:
        Validated instance of response_model.

    Raises:
        StructuredOutputError: If all retries fail.
    """
    tool_schema = _pydantic_to_tool_schema(response_model)
    tool_name = tool_schema["function"]["name"]

    last_error = None

    for attempt in range(1 + max_retries):
        try:
            # Attempt 1: Tool calling (preferred)
            if attempt <= max_retries // 2 + 1:
                result = await _try_tool_calling(
                    model, messages, tool_schema, tool_name, response_model, temperature
                )
            else:
                # Fallback: JSON mode
                result = await _try_json_mode(model, messages, response_model, temperature)
            return result

        except (json.JSONDecodeError, ValidationError, KeyError, IndexError, TypeError) as e:
            last_error = e
            logger.warning(f"Structured output attempt {attempt + 1}/{1 + max_retries} failed: {e}")
            continue

        except Exception as e:
            last_error = e
            logger.warning(f"Structured output attempt {attempt + 1}/{1 + max_retries} error: {e}")
            # On first failure, try JSON mode fallback (covers tool calling
            # not supported, Azure API errors, etc.)
            if attempt == 0:
                logger.info(f"Tool calling failed for {model}, trying JSON mode fallback")
                try:
                    return await _try_json_mode(model, messages, response_model, temperature)
                except Exception as fallback_e:
                    last_error = fallback_e
                    logger.warning(f"JSON mode fallback also failed: {fallback_e}")
            continue

    raise StructuredOutputError(
        f"Failed to get structured output after {1 + max_retries} attempts. "
        f"Model: {model}, Schema: {response_model.__name__}. "
        f"Last error: {last_error}"
    )


async def _try_tool_calling(
    model: str,
    messages: list[dict],
    tool_schema: dict,
    tool_name: str,
    response_model: type[T],
    temperature: float,
) -> T:
    """Attempt structured output via forced tool calling."""
    response = await acompletion(
        model=model,
        messages=messages,
        tools=[tool_schema],
        tool_choice={"type": "function", "function": {"name": tool_name}},
        temperature=temperature,
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        raise ValueError("LLM did not return a tool call")

    arguments_str = tool_calls[0].function.arguments
    arguments = json.loads(arguments_str)
    return response_model.model_validate(arguments)


async def _try_json_mode(
    model: str,
    messages: list[dict],
    response_model: type[T],
    temperature: float,
) -> T:
    """Fallback: attempt structured output via JSON mode."""
    # Add schema hint to the last message
    schema_hint = (
        f"\n\nYou MUST respond with valid JSON matching this exact schema:\n"
        f"```json\n{json.dumps(response_model.model_json_schema(), indent=2)}\n```"
    )
    augmented_messages = messages.copy()
    if augmented_messages:
        last_msg = augmented_messages[-1].copy()
        last_msg["content"] = last_msg.get("content", "") + schema_hint
        augmented_messages[-1] = last_msg

    response = await acompletion(
        model=model,
        messages=augmented_messages,
        response_format={"type": "json_object"},
        temperature=temperature,
    )

    text = response.choices[0].message.content
    data = json.loads(text)
    return response_model.model_validate(data)


async def call_llm_plain(
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
) -> str:
    """Call an LLM and return the plain text response.

    Use this for target model calls where we want natural, unstructured output.

    Args:
        model: LiteLLM model string.
        messages: Chat messages to send.
        temperature: LLM temperature.

    Returns:
        The LLM's text response.
    """
    response = await acompletion(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


class StructuredOutputError(Exception):
    """Raised when structured output extraction fails after all retries."""

    pass
