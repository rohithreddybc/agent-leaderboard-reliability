"""Minimal Groq chat-completions client with rate-limit back-off.

The API key is read from the GROQ_API_KEY environment variable at call time. It is
never logged, never written to disk, and never included in exceptions.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Callable

API_URL = "https://api.groq.com/openai/v1/chat/completions"


class QuotaExhausted(Exception):
    """A per-day limit (tokens or requests) was hit. Checkpoint and stop; resume later."""


class ApiError(Exception):
    def __init__(self, kind: str, detail: str, status: int | None = None, body: Any = None):
        super().__init__(f"{kind}: {detail}")
        self.kind, self.detail, self.status, self.body = kind, detail, status, body


def parse_duration(text: str | None) -> float | None:
    """Parse '1m26.4s', '3.195s', '250ms', '2h3m1s' into seconds."""
    if not text:
        return None
    total, found = 0.0, False
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", text):
        found = True
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total if found else None


def _default_transport(url: str, headers: dict, payload: dict, timeout: float):
    import requests

    return requests.post(url, headers=headers, json=payload, timeout=timeout)


class GroqClient:
    def __init__(
        self,
        transport: Callable | None = None,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 60.0,
        max_rate_retries: int = 30,
        max_transient_retries: int = 2,
        min_tokens_headroom: int = 3000,
        api_key: str | None = None,
        log: Callable[[str], None] = lambda s: None,
    ):
        self.transport = transport or _default_transport
        self.sleep = sleep
        self.timeout = timeout
        self.max_rate_retries = max_rate_retries
        self.max_transient_retries = max_transient_retries
        self.min_tokens_headroom = min_tokens_headroom
        self._api_key = api_key
        self.log = log
        self._remaining_tokens: int | None = None
        self._reset_tokens_s: float = 0.0
        self.events: list[str] = []  # per-call retry/error events, drained by the caller

    def _headers(self) -> dict:
        key = self._api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ApiError("no_api_key", "GROQ_API_KEY is not set")
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _note_headers(self, resp) -> None:
        h = {k.lower(): v for k, v in getattr(resp, "headers", {}).items()}
        rem = h.get("x-ratelimit-remaining-tokens")
        if rem is not None:
            try:
                self._remaining_tokens = int(float(rem))
            except ValueError:
                self._remaining_tokens = None
        self._reset_tokens_s = parse_duration(h.get("x-ratelimit-reset-tokens")) or 0.0

    def chat(self, payload: dict) -> tuple[dict, float, int]:
        """POST one chat completion. Returns (json, seconds_for_successful_request, retries).

        Raises QuotaExhausted on a per-day limit, ApiError on any other unrecovered failure.
        A 400 with code 'tool_use_failed' is raised as ApiError(kind='tool_use_failed').
        """
        retries = 0
        rate_tries = 0
        transient = 0
        while True:
            if self._remaining_tokens is not None and self._remaining_tokens < self.min_tokens_headroom:
                self.sleep(min(self._reset_tokens_s, 65.0) + 0.5)
                self._remaining_tokens = None
            t0 = time.perf_counter()
            try:
                resp = self.transport(API_URL, self._headers(), payload, self.timeout)
            except ApiError:
                raise
            except Exception as e:  # timeouts, connection resets
                kind = "timeout" if "imeout" in type(e).__name__ else "connection_error"
                transient += 1
                self.events.append(f"{kind}")
                if transient > self.max_transient_retries:
                    raise ApiError(kind, type(e).__name__) from None
                retries += 1
                self.sleep(2.0 * transient)
                continue
            dt = time.perf_counter() - t0
            self._note_headers(resp)
            status = resp.status_code
            if status == 200:
                return resp.json(), dt, retries
            try:
                body = resp.json()
            except Exception:
                body = {"error": {"message": (getattr(resp, "text", "") or "")[:300]}}
            err = body.get("error", {}) if isinstance(body, dict) else {}
            msg = str(err.get("message", ""))
            code = err.get("code")
            if status == 429:
                if re.search(r"per day|\bTPD\b|\bRPD\b", msg, re.IGNORECASE):
                    raise QuotaExhausted(msg[:300])
                rate_tries += 1
                self.events.append("rate_limited_429")
                if rate_tries > self.max_rate_retries:
                    raise ApiError("rate_limit_persistent", msg[:200], status, body)
                retries += 1
                h = {k.lower(): v for k, v in resp.headers.items()}
                wait = parse_duration(h.get("retry-after")) or parse_duration(
                    (re.search(r"try again in ([\dhms.]+)", msg) or [None, None])[1]
                ) or 10.0
                self.sleep(min(wait, 120.0) + 1.0)
                self._remaining_tokens = None
                continue
            if status == 400 and code == "tool_use_failed":
                raise ApiError("tool_use_failed", msg[:300], status, body)
            if status >= 500:
                transient += 1
                self.events.append(f"http_{status}")
                if transient > self.max_transient_retries:
                    raise ApiError(f"http_{status}", msg[:200], status, body)
                retries += 1
                self.sleep(3.0 * transient)
                continue
            raise ApiError(f"http_{status}", msg[:300], status, body)
