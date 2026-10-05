"""Tool-use agent loop and per-run record construction."""
from __future__ import annotations

import datetime as dt
import time

from client import ApiError, GroqClient
from tasks import SYSTEM_PROMPT, Task, check_success
from world import TOOL_SCHEMAS, World, validate_call

MAX_STEPS = 10  # maximum number of model calls per run
MAX_COMPLETION_TOKENS = 2048


def run_one(
    client: GroqClient,
    model: str,
    task: Task,
    repeat: int,
    condition: str,
    temperature: float | None,
    max_steps: int = MAX_STEPS,
) -> dict:
    """Run one task once. Always returns a record, including for failed runs.

    QuotaExhausted is deliberately not caught here: a run abandoned by a daily-quota
    stop is not a completed run, so the caller checkpoints and resumes it later.
    """
    world = World()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    calls: list[dict] = []
    invalid_breakdown: dict[str, int] = {}
    pt = ct = tt = 0
    api_latency = 0.0
    retries = 0
    steps = 0
    error: str | None = None
    error_detail: str | None = None
    stop_reason = "max_steps"
    final_answer: str | None = None
    client.events = []
    started = dt.datetime.now(dt.timezone.utc)
    t_wall = time.perf_counter()

    def bump(cat: str) -> None:
        invalid_breakdown[cat] = invalid_breakdown.get(cat, 0) + 1

    while steps < max_steps:
        payload = {
            "model": model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        steps += 1
        try:
            data, secs, r = client.chat(payload)
        except ApiError as e:
            if e.kind == "tool_use_failed":
                bump("api_tool_use_failed")
                calls.append({"name": None, "valid": False, "category": "api_tool_use_failed"})
                messages.append(
                    {"role": "user", "content": "Your last tool call was malformed and could not be "
                     "parsed. Call the tool again with valid JSON arguments, or give your final answer."}
                )
                continue
            error, error_detail, stop_reason = e.kind, e.detail[:200], "error"
            break
        retries += r
        api_latency += secs
        usage = data.get("usage") or {}
        pt += usage.get("prompt_tokens", 0) or 0
        ct += usage.get("completion_tokens", 0) or 0
        tt += usage.get("total_tokens", 0) or 0
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        assistant = {"role": "assistant", "content": msg.get("content")}
        if tool_calls:
            assistant["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or "{}"}}
                for tc in tool_calls
            ]
        messages.append(assistant)
        if not tool_calls:
            final_answer = (msg.get("content") or "").strip()
            stop_reason = "length" if choice.get("finish_reason") == "length" else (
                "final_answer" if final_answer else "empty_reply")
            break
        for tc in tool_calls:
            name = tc["function"]["name"]
            raw = tc["function"].get("arguments") or "{}"
            args, cat, why = validate_call(name, raw)
            if cat is not None:
                bump(cat)
                calls.append({"name": name, "args": str(raw)[:200], "valid": False, "category": cat})
                result = f"ERROR: invalid tool call: {why}"
            else:
                text, tool_err = world.call(name, args)
                calls.append({"name": name, "args": args, "valid": True, "tool_error": tool_err})
                result = text
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    else:
        stop_reason = "max_steps"

    success = check_success(task, world.state, final_answer) if error is None else False
    n_invalid = sum(1 for c in calls if not c["valid"])
    return {
        "condition": condition,
        "model": model,
        "date": started.strftime("%Y-%m-%d"),
        "timestamp_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task_id": task.id,
        "repeat": repeat,
        "temperature": temperature,
        "temperature_setting": "provider_default" if temperature is None else str(temperature),
        "max_steps": max_steps,
        "success": bool(success),
        "steps": steps,
        "tool_calls": len(calls),
        "invalid_tool_calls": n_invalid,
        "invalid_breakdown": invalid_breakdown,
        "tool_errors": sum(1 for c in calls if c["valid"] and c.get("tool_error")),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
        "latency_s": round(api_latency, 3),
        "wall_s": round(time.perf_counter() - t_wall, 3),
        "retries": retries,
        "retry_events": list(client.events),
        "error": error,
        "error_detail": error_detail,
        "stop_reason": stop_reason,
        "final_answer": (final_answer or "")[:500],
        "calls": calls,
    }
