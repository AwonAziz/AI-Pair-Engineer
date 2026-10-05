"""OpenRouter client and structured-output parsing.

Two deliberate design choices:

1. The API key is resolved lazily, on first use. Raising at import time means a
   misconfigured app cannot even start far enough to tell you the key is
   missing, and it makes the package unimportable for anyone without a key.
2. Transport failures and contract failures are separate exception types.
   Callers retry the former and treat the latter as a correctness problem.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APIStatusError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class TokenUsage(BaseModel):
    """Measured token consumption for one provider call."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-chat"

# Published per-million-token prices, used only to turn measured token counts
# into a dollar figure in benchmark reports. These drift, so treat the cost
# column as an estimate and the token counts as the real measurement. Anything
# unpriced reports zero rather than a guess.
USD_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-sonnet-4": (3.00, 15.00),
    "anthropic/claude-opus-4": (15.00, 75.00),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-2.5-pro": (1.25, 10.00),
    "meta-llama/llama-3.3-70b-instruct": (0.12, 0.30),
    "deepseek/deepseek-chat": (0.27, 1.10),
    "qwen/qwen-2.5-72b-instruct": (0.12, 0.39),
}

# Exponential backoff ceiling. Without a cap, a persistently rate-limited key
# produces multi-minute sleeps inside a Streamlit request handler.
MAX_BACKOFF_SECONDS = 30.0

ModelT = TypeVar("ModelT", bound=BaseModel)

load_dotenv()

_client: OpenAI | None = None


class LLMError(RuntimeError):
    """Base class for every failure originating from this module."""


class MissingAPIKeyError(LLMError):
    """No API key was configured."""


class LLMTransportError(LLMError):
    """The request failed for a reason that may succeed on retry."""


class LLMResponseError(LLMError):
    """The response arrived but did not satisfy the requested contract."""


def is_configured() -> bool:
    """Whether an API key is available, without raising."""
    return bool(os.getenv("OPENROUTER_API_KEY", "").strip())


def get_client() -> OpenAI:
    """Return a lazily constructed OpenAI-compatible client.

    Raises:
        MissingAPIKeyError: if ``OPENROUTER_API_KEY`` is unset or blank.
    """
    global _client

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise MissingAPIKeyError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add "
            "your key, or export it in your shell."
        )

    if _client is None:
        _client = OpenAI(
            base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).strip(),
            api_key=api_key,
            timeout=float(os.getenv("OPENROUTER_TIMEOUT", "120")),
            max_retries=0,  # retries are handled here so backoff stays visible
        )
    return _client


def reset_client() -> None:
    """Drop the cached client. Used by tests and after a key change."""
    global _client
    _client = None


def ask_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    max_retries: int = 3,
    on_usage: Callable[[TokenUsage], None] | None = None,
) -> str:
    """Send one chat completion and return the raw assistant text.

    Retries transport failures with exponential backoff plus jitter. The
    assistant's content is never logged or included in error messages, since
    it embeds the user's submitted source code.

    ``on_usage`` is called with the token counts for each attempt that reached
    the provider, so a caller can total real spend. A benchmark cannot report a
    cost it did not measure.

    Raises:
        MissingAPIKeyError: no API key configured.
        LLMTransportError: every attempt failed.
    """
    client = get_client()
    resolved_model = model or os.getenv("MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    # Typed as the concrete param type so the SDK's overloads accept it.
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                if on_usage is not None:
                    on_usage(_read_usage(response, resolved_model))
                return content
            last_error = LLMTransportError("provider returned an empty completion")
            logger.warning("Empty completion on attempt %s/%s", attempt, max_retries)

        except RateLimitError as exc:
            last_error = exc
            logger.warning("Rate limited on attempt %s/%s", attempt, max_retries)

        except APIStatusError as exc:
            last_error = exc
            # 4xx other than 429 will not become valid by repeating the request.
            if exc.status_code < 500 and exc.status_code != 429:
                raise LLMTransportError(
                    f"provider rejected the request (HTTP {exc.status_code})"
                ) from exc
            logger.warning("Provider error %s on attempt %s", exc.status_code, attempt)

        except (APIConnectionError, APIError) as exc:
            last_error = exc
            logger.warning("Transport error on attempt %s/%s", attempt, max_retries)

        if attempt < max_retries:
            rate_limited = isinstance(last_error, RateLimitError)
            time.sleep(_backoff_seconds(attempt, rate_limited=rate_limited))

    raise LLMTransportError(
        f"LLM request failed after {max_retries} attempt(s): {_describe(last_error)}"
    ) from last_error


def _read_usage(response: Any, model: str) -> TokenUsage:
    """Extract token counts from a provider response, tolerating absence.

    Not every OpenRouter-compatible endpoint reports usage, and a model routed
    to a non-OpenAI backend may omit it. A missing count becomes zero rather
    than an exception, so an unpriced run still reports what it knows.
    """
    raw = getattr(response, "usage", None)
    prompt_tokens = int(getattr(raw, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(raw, "completion_tokens", 0) or 0)
    return TokenUsage(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def estimate_cost_usd(usage: TokenUsage) -> float:
    """Estimate the dollar cost of one call from measured tokens.

    Longest-prefix match on the model id, so provider-prefixed ids like
    ``anthropic/claude-sonnet-4-20250514`` still resolve. Returns 0.0 for an
    unpriced model instead of inventing a figure.
    """
    pricing = _lookup_pricing(usage.model)
    if pricing is None:
        return 0.0
    input_price, output_price = pricing
    return (usage.prompt_tokens * input_price + usage.completion_tokens * output_price) / 1_000_000


def _lookup_pricing(model: str) -> tuple[float, float] | None:
    normalized = model.strip().lower()
    best: tuple[str, tuple[float, float]] | None = None
    for prefix, pricing in USD_PER_MILLION_TOKENS.items():
        if normalized.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, pricing)
    return best[1] if best else None


def _backoff_seconds(attempt: int, *, rate_limited: bool) -> float:
    """Exponential backoff with full jitter, capped.

    Jitter matters when several pipeline stages fail at once: without it every
    retry lands on the same tick and re-creates the burst that caused the limit.
    """
    ceiling = min(MAX_BACKOFF_SECONDS, 2.0**attempt * (5.0 if rate_limited else 1.0))
    return random.uniform(0.0, ceiling)


def _describe(error: Exception | None) -> str:
    """Summarise an error without leaking key material."""
    if error is None:
        return "unknown error"
    message = str(error).replace(os.getenv("OPENROUTER_API_KEY", "") or "\x00", "")
    return f"{type(error).__name__}: {message[:300]}"


def parse_structured_response(raw_response: str, schema_type: type[ModelT]) -> ModelT:
    """Parse an assistant response into ``schema_type``.

    Handles the three shapes models actually emit: bare JSON, JSON inside a
    ```` ```json ```` fence, and JSON surrounded by prose.

    Raises:
        LLMResponseError: if no JSON object could be recovered, or if the
            recovered JSON does not satisfy the schema.
    """
    for candidate in _json_candidates(raw_response):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(parsed, dict):
            continue

        try:
            return schema_type.model_validate(parsed)
        except ValidationError as exc:
            raise LLMResponseError(_format_validation_error(exc)) from exc

    raise LLMResponseError(
        f"could not recover a JSON object matching {schema_type.__name__} from the response"
    )


def _json_candidates(raw_response: str) -> list[str]:
    """Yield increasingly permissive candidate JSON strings."""
    candidates: list[str] = []

    fenced = _extract_fenced_block(raw_response)
    if fenced is not None:
        candidates.append(fenced)

    balanced = _extract_balanced_object(raw_response)
    if balanced is not None:
        candidates.append(balanced)

    stripped = raw_response.strip()
    if stripped:
        candidates.append(stripped)

    return candidates


def _extract_fenced_block(text: str) -> str | None:
    """Return the contents of the first fenced code block, or None.

    Accepts ```` ```json ````, a bare ```` ``` ```` fence, and unterminated
    fences, which happen whenever a response is cut off by max_tokens.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip().startswith("```"):
            continue
        body: list[str] = []
        for candidate in lines[index + 1 :]:
            if candidate.strip().startswith("```"):
                break
            body.append(candidate)
        else:
            # Unterminated fence: the block ran to the end of the response.
            if body:
                return "\n".join(body)
        if body:
            return "\n".join(body)
    return None


def _extract_balanced_object(text: str) -> str | None:
    """Slice out the first brace-balanced ``{...}`` region.

    This is string-aware: braces inside string literals are skipped, so JSON
    containing code snippets with unbalanced braces still parses.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _format_validation_error(exc: ValidationError) -> str:
    """Render a Pydantic error compactly for logging and exception text."""
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        problems.append(f"{location}: {error['msg']}")
    return f"schema validation failed: {'; '.join(problems[:5])}"


def extract_response_text(response: Any) -> str:
    """Best-effort extraction of assistant text from a provider response."""
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError):
        return ""
