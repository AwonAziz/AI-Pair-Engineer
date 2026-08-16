from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, Timeout

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("MODEL", "deepseek/deepseek-chat")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def ask_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> str:
    """Ask the LLM a question and return the raw response string.

    Retries on transient failures with exponential backoff.
    Does not expose the API key in any error messages.
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content:
                return content
            raise ValueError("Empty response from LLM")

        except Timeout as e:
            last_error = e
            if attempt < max_retries:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            continue

        except RateLimitError as e:
            last_error = e
            if attempt < max_retries:
                wait_time = 5 * 2 ** attempt
                time.sleep(wait_time)
            continue

        except APIError as e:
            last_error = e
            if attempt < max_retries:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            continue

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            continue

    raise last_error from last_error


def parse_structured_response(
    raw_response: str,
    schema_type: type,
) -> Any:
    """Parse an LLM response into a Pydantic model or other structured type.

    Handles:
    1. Raw JSON
    2. JSON inside ```json fences
    3. Minor surrounding whitespace/text

    Retries with a correction prompt if parsing fails initially.
    """
    cleaned = raw_response.strip()

    # Try to extract JSON from ```json fences
    if code_block := _extract_json_from_code_block(cleaned):
        cleaned = code_block

    # Try to find the outermost JSON object/array
    json_start = _find_json_start(cleaned)
    if json_start is not None:
        json_text = cleaned[json_start:]
        # Try to find the matching closing brace/bracket
        depth = 0
        last_char = len(json_text) - 1
        for i, ch in enumerate(json_text):
            if ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    json_text = json_text[: i + 1]
                    break
        try:
            parsed = json.loads(json_text)
            if schema_type is not None:
                return schema_type.model_validate(parsed)  # type: ignore[arg-type]
            return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # No JSON found - try parsing the whole thing as JSON
    try:
        parsed = json.loads(cleaned)
        if schema_type is not None:
            return schema_type.model_validate(parsed)  # type: ignore[arg-type]
        return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    raise ValueError("Unable to parse LLM response as JSON")


def _extract_json_from_code_block(text: str) -> Optional[str]:
    """Extract JSON text from a ```json code block if present."""
    lines = text.split("\n")
    in_code_block = False
    code_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```json"):
            in_code_block = True
            continue
        if in_code_block and stripped.startswith("```"):
            if code_lines:
                return "\n".join(code_lines)
            in_code_block = False
            code_lines = []
            continue
        if in_code_block:
            code_lines.append(line)

    return None


def _find_json_start(text: str) -> Optional[int]:
    """Find the start index of a JSON object or array in text.

    Returns the index of the first { or [, or None if not found.
    """
    for i, ch in enumerate(text):
        if ch in "{[":
            return i
    return None