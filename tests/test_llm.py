"""Tests for the OpenRouter client wrapper.

The bugs worth guarding against here are operational: leaking the key into an
error message, retrying a request that cannot succeed, and sleeping for an
unbounded time inside a request handler.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from openai import APIStatusError, RateLimitError

from ai_pair_engineer.services import llm as llm_module
from ai_pair_engineer.services.llm import (
    LLMTransportError,
    MissingAPIKeyError,
    TokenUsage,
    ask_llm,
    estimate_cost_usd,
    get_client,
    is_configured,
    reset_client,
)

SYSTEM = "system"
USER = "user"


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Replace the OpenAI constructor with a recording fake.

    Yields a single-element list that is populated once ``get_client`` is
    first called, so a test can reach into ``calls`` afterwards.
    """
    holder: dict[str, Any] = {
        "behaviour": lambda _n: _completion("ok"),
        "constructions": 0,
    }

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return holder["behaviour"](len(self.calls))

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            holder["constructions"] += 1
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm_module, "_client", None)
    monkeypatch.setattr(llm_module, "OpenAI", FakeClient)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-value")
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)

    holder["client"] = lambda: get_client()
    yield holder
    llm_module.reset_client()


def _record_usage() -> list[TokenUsage]:
    """Build an ``on_usage`` callback that appends to a list it returns."""
    recorded: list[TokenUsage] = []
    return recorded


def _completion_without_usage(content: str) -> Any:
    """A response shaped like a provider that does not report usage."""
    message = type("Message", (), {"content": content})()
    choice = type("Choice", (), {"message": message})()
    return type("Response", (), {"choices": [choice]})()


def _client(fake_client: dict[str, Any]) -> Any:
    return fake_client["client"]()


def _completion(
    content: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Any:
    message = type("Message", (), {"content": content})()
    choice = type("Choice", (), {"message": message})()
    usage = type(
        "Usage",
        (),
        {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    )()
    return type("Response", (), {"choices": [choice], "usage": usage})()


class TestConfiguration:
    def test_is_configured_is_false_without_a_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert not is_configured()

    def test_is_configured_is_false_for_a_blank_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
        assert not is_configured()

    def test_is_configured_is_true_with_a_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        assert is_configured()

    def test_get_client_raises_a_dedicated_error_without_a_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        reset_client()
        with pytest.raises(MissingAPIKeyError, match="OPENROUTER_API_KEY"):
            get_client()

    def test_importing_the_module_does_not_require_a_key(self) -> None:
        """Importing must not raise when no key is configured.

        The original module did ``raise ValueError`` at import time, which meant
        the Streamlit app could not start far enough to tell you the key was
        missing, and the parser was unimportable for anyone without one.

        Run in a subprocess because ``importlib.reload`` would replace the
        exception classes, breaking ``pytest.raises`` in every later test.
        """
        code = "import ai_pair_engineer.services.llm as m;print('imported', m.is_configured())"
        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert "imported False" in completed.stdout

    def test_the_client_is_constructed_once(self, fake_client: dict[str, Any]) -> None:
        get_client()
        get_client()
        assert fake_client["constructions"] == 1

    def test_reset_client_forces_a_rebuild(self, fake_client: dict[str, Any]) -> None:
        get_client()
        reset_client()
        get_client()
        assert fake_client["constructions"] == 2

    def test_the_key_is_never_in_an_error_message(
        self, monkeypatch: pytest.MonkeyPatch, fake_client: dict[str, Any]
    ) -> None:
        """A leaked key in a log line or traceback is a credential leak."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "super-secret-key")
        fake_client["behaviour"] = lambda n: _raise(
            APIStatusError(
                "rejected: super-secret-key",
                response=_response(401),
                body=None,
            )
        )

        with pytest.raises(LLMTransportError) as info:
            ask_llm(SYSTEM, USER, max_retries=2)

        assert "super-secret-key" not in str(info.value)


class TestSuccessfulRequests:
    def test_returns_the_assistant_text(self, fake_client: dict[str, Any]) -> None:
        _set_behaviour(fake_client, lambda n: _completion("hello"))
        assert ask_llm(SYSTEM, USER) == "hello"

    def test_sends_system_and_user_messages(self, fake_client: dict[str, Any]) -> None:
        _set_behaviour(fake_client, lambda n: _completion("ok"))
        ask_llm(SYSTEM, USER)
        sent = _client(fake_client).chat.completions.calls[0]["messages"]
        assert sent == [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ]

    def test_uses_the_default_temperature(self, fake_client: dict[str, Any]) -> None:
        _set_behaviour(fake_client, lambda n: _completion("ok"))
        ask_llm(SYSTEM, USER)
        assert _client(fake_client).chat.completions.calls[0]["temperature"] == 0.1

    def test_honours_an_explicit_temperature(self, fake_client: dict[str, Any]) -> None:
        _set_behaviour(fake_client, lambda n: _completion("ok"))
        ask_llm(SYSTEM, USER, temperature=0.0)
        assert _client(fake_client).chat.completions.calls[0]["temperature"] == 0.0

    def test_uses_the_configured_model(
        self, fake_client: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODEL", "vendor/some-model")
        _set_behaviour(fake_client, lambda n: _completion("ok"))
        ask_llm(SYSTEM, USER)
        assert _client(fake_client).chat.completions.calls[0]["model"] == "vendor/some-model"

    def test_an_explicit_model_beats_the_environment(
        self, fake_client: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODEL", "env/model")
        _set_behaviour(fake_client, lambda n: _completion("ok"))
        ask_llm(SYSTEM, USER, model="explicit/model")
        assert _client(fake_client).chat.completions.calls[0]["model"] == "explicit/model"

    def test_usage_is_not_recorded_without_a_callback(self, fake_client: dict[str, Any]) -> None:
        _set_behaviour(
            fake_client, lambda n: _completion("ok", prompt_tokens=10, completion_tokens=5)
        )
        ask_llm(SYSTEM, USER)  # must not raise without a callback


class TestTokenAccounting:
    def test_usage_is_reported_to_the_callback(self, fake_client: dict[str, Any]) -> None:
        _set_behaviour(
            fake_client, lambda n: _completion("ok", prompt_tokens=120, completion_tokens=45)
        )
        recorded: list[TokenUsage] = []
        ask_llm(SYSTEM, USER, on_usage=recorded.append)
        assert len(recorded) == 1
        assert recorded[0].prompt_tokens == 120
        assert recorded[0].completion_tokens == 45
        assert recorded[0].total_tokens == 165

    def test_recorded_usage_carries_the_resolved_model(self, fake_client: dict[str, Any]) -> None:
        _set_behaviour(fake_client, lambda n: _completion("ok"))
        recorded: list[TokenUsage] = []
        ask_llm(SYSTEM, USER, model="vendor/some-model", on_usage=recorded.append)
        assert recorded[0].model == "vendor/some-model"

    def test_a_response_without_usage_yields_zeros(self, fake_client: dict[str, Any]) -> None:
        """Not every OpenRouter-compatible backend reports usage."""
        _set_behaviour(fake_client, lambda n: _completion_without_usage("ok"))
        recorded: list[TokenUsage] = []
        ask_llm(SYSTEM, USER, on_usage=recorded.append)
        assert recorded[0].total_tokens == 0

    def test_only_the_successful_attempt_is_reported(self, fake_client: dict[str, Any]) -> None:
        _set_behaviour(
            fake_client,
            lambda n: (
                _completion("", prompt_tokens=1)
                if n == 1
                else _completion("ok", prompt_tokens=2, completion_tokens=3)
            ),
        )
        recorded: list[TokenUsage] = []
        ask_llm(SYSTEM, USER, max_retries=3, on_usage=recorded.append)
        # The empty completion produced no content, so it was not billed.
        assert len(recorded) == 1
        assert recorded[0].total_tokens == 5


class TestRetries:
    def test_retries_an_empty_completion(self, fake_client: dict[str, Any]) -> None:
        _set_behaviour(fake_client, lambda n: _completion("") if n == 1 else _completion("ok"))
        assert ask_llm(SYSTEM, USER, max_retries=3) == "ok"

    def test_retries_a_rate_limit(self, fake_client: dict[str, Any]) -> None:
        _set_behaviour(
            fake_client,
            lambda n: (
                _raise(RateLimitError("slow down", response=_response(429), body=None))
                if n == 1
                else _completion("ok")
            ),
        )
        assert ask_llm(SYSTEM, USER, max_retries=3) == "ok"

    def test_stops_after_max_retries(self, fake_client: dict[str, Any]) -> None:
        _set_behaviour(fake_client, lambda n: _completion(""))
        with pytest.raises(LLMTransportError, match="after 3 attempt"):
            ask_llm(SYSTEM, USER, max_retries=3)

    def test_does_not_retry_a_client_error(self, fake_client: dict[str, Any]) -> None:
        """A 401 will still be a 401 on the third try."""
        _set_behaviour(
            fake_client,
            lambda n: _raise(APIStatusError("unauthorized", response=_response(401), body=None)),
        )
        with pytest.raises(LLMTransportError, match="401"):
            ask_llm(SYSTEM, USER, max_retries=3)

    def test_retries_a_server_error(self, fake_client: dict[str, Any]) -> None:
        _set_behaviour(
            fake_client,
            lambda n: (
                _raise(APIStatusError("bad gateway", response=_response(502), body=None))
                if n < 3
                else _completion("ok")
            ),
        )
        assert ask_llm(SYSTEM, USER, max_retries=3) == "ok"

    def test_error_message_reports_the_attempt_count(self, fake_client: dict[str, Any]) -> None:
        _set_behaviour(fake_client, lambda n: _completion(""))
        with pytest.raises(LLMTransportError, match="after 2 attempt"):
            ask_llm(SYSTEM, USER, max_retries=2)


class TestCostEstimation:
    def test_computes_from_measured_tokens(self) -> None:
        usage = TokenUsage(
            model="anthropic/claude-sonnet-4", prompt_tokens=1_000_000, completion_tokens=0
        )
        assert estimate_cost_usd(usage) == pytest.approx(3.00)

    def test_prices_output_separately(self) -> None:
        usage = TokenUsage(
            model="anthropic/claude-sonnet-4", prompt_tokens=0, completion_tokens=1_000_000
        )
        assert estimate_cost_usd(usage) == pytest.approx(15.00)

    def test_prices_a_split_call(self) -> None:
        usage = TokenUsage(
            model="anthropic/claude-sonnet-4",
            prompt_tokens=500_000,
            completion_tokens=100_000,
        )
        assert estimate_cost_usd(usage) == pytest.approx(1.50 + 1.50)

    def test_matches_a_dated_model_id(self) -> None:
        """OpenRouter ids carry a date suffix; the prefix must still resolve."""
        usage = TokenUsage(model="anthropic/claude-sonnet-4-20250514", prompt_tokens=1_000_000)
        assert estimate_cost_usd(usage) > 0

    def test_prefers_the_most_specific_prefix(self) -> None:
        usage = TokenUsage(model="anthropic/claude-3.5-sonnet", prompt_tokens=1_000_000)
        assert estimate_cost_usd(usage) == pytest.approx(3.00)

    def test_unpriced_model_reports_zero_rather_than_guessing(self) -> None:
        usage = TokenUsage(model="some-local-model", prompt_tokens=1_000_000)
        assert estimate_cost_usd(usage) == 0.0

    def test_unknown_model_reports_zero(self) -> None:
        usage = TokenUsage(model="", prompt_tokens=500)
        assert estimate_cost_usd(usage) == 0.0


class TestBackoff:
    def test_backoff_grows_with_the_attempt(self) -> None:
        first = llm_module._backoff_seconds(1, rate_limited=False)
        third = llm_module._backoff_seconds(3, rate_limited=False)
        # Full jitter makes single samples noisy, so compare the ceilings.
        assert max(first, third) <= llm_module.MAX_BACKOFF_SECONDS

    def test_backoff_is_capped(self) -> None:
        for attempt in range(1, 30):
            assert llm_module._backoff_seconds(attempt, rate_limited=True) <= (
                llm_module.MAX_BACKOFF_SECONDS
            )

    def test_rate_limited_backoff_is_longer_on_average(self) -> None:
        """A rate limit needs a longer pause than a generic error."""
        normal = [llm_module._backoff_seconds(3, rate_limited=False) for _ in range(200)]
        limited = [llm_module._backoff_seconds(3, rate_limited=True) for _ in range(200)]
        assert sum(limited) / len(limited) > sum(normal) / len(normal)

    def test_backoff_includes_jitter(self) -> None:
        """Identical delays would recreate the burst that caused the limit."""
        samples = {llm_module._backoff_seconds(4, rate_limited=True) for _ in range(50)}
        assert len(samples) > 1


def _set_behaviour(fake_client: dict[str, Any], behaviour: Any) -> None:
    fake_client["behaviour"] = behaviour


def _raise(exc: Exception) -> Any:
    raise exc


def _response(status_code: int) -> Any:
    return type("Response", (), {"status_code": status_code, "headers": {}, "request": None})()
