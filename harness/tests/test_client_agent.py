import pytest

from agent import run_one
from client import ApiError, GroqClient, QuotaExhausted, parse_duration
from fakes import FakeResp, Scripted, completion
from tasks import TASKS_BY_ID


def NOSLEEP(s):
    return None


def mk(items, **kw):
    t = Scripted(items)
    return GroqClient(transport=t, sleep=NOSLEEP, api_key="test-key-not-real", **kw), t


def test_parse_duration():
    assert parse_duration("1m26.4s") == pytest.approx(86.4)
    assert parse_duration("3.195s") == pytest.approx(3.195)
    assert parse_duration("250ms") == pytest.approx(0.25)
    assert parse_duration("2h3m1s") == 7381
    assert parse_duration(None) is None and parse_duration("soon") is None


def test_429_backs_off_then_succeeds():
    c, t = mk([FakeResp(429, {"error": {"message": "Rate limit ... tokens per minute (TPM). try again in 2s"}},
                        {"retry-after": "2"}), completion("hi")])
    data, _, retries = c.chat({"model": "m"})
    assert retries == 1 and "choices" in data and c.events == ["rate_limited_429"]


def test_daily_quota_raises():
    c, _ = mk([FakeResp(429, {"error": {"message": "Rate limit reached on tokens per day (TPD): Limit 200000"}})])
    with pytest.raises(QuotaExhausted):
        c.chat({})


def test_timeout_retried_then_error():
    class ReadTimeout(Exception):
        pass

    c, _ = mk([ReadTimeout(), ReadTimeout(), ReadTimeout()])
    with pytest.raises(ApiError) as e:
        c.chat({})
    assert e.value.kind == "timeout"


def test_missing_key_is_an_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    c = GroqClient(transport=Scripted([]), sleep=NOSLEEP)
    with pytest.raises(ApiError):
        c.chat({})


def test_run_success_path_and_accounting():
    task = TASKS_BY_ID["t03"]
    c, t = mk([
        completion(tool_calls=[("place_order", {"customer": "alice", "item": "widget", "qty": 3})], pt=200, ct=20),
        completion("Done.", pt=250, ct=5),
    ])
    rec = run_one(c, "m", task, 1, "default", None)
    assert rec["success"] and rec["steps"] == 2 and rec["tool_calls"] == 1
    assert rec["invalid_tool_calls"] == 0 and rec["total_tokens"] == 475
    assert "temperature" not in t.payloads[0]  # provider default: parameter omitted
    assert rec["temperature_setting"] == "provider_default" and rec["error"] is None


def test_temperature_zero_is_sent():
    c, t = mk([completion("12")])
    rec = run_one(c, "m", TASKS_BY_ID["t01"], 1, "temp0", 0.0)
    assert t.payloads[0]["temperature"] == 0.0 and rec["temperature_setting"] == "0.0"


def test_invalid_calls_counted_and_recoverable():
    task = TASKS_BY_ID["t06"]
    c, _ = mk([
        completion(tool_calls=[("cancel_order", "{bad json"), ("cancel_ordr", {"order_id": "O-1002"}),
                               ("cancel_order", {"order_id": 1002})]),
        completion(tool_calls=[("cancel_order", {"order_id": "O-1002"})]),
        completion("Cancelled."),
    ])
    rec = run_one(c, "m", task, 1, "default", None)
    assert rec["success"]
    assert rec["tool_calls"] == 4 and rec["invalid_tool_calls"] == 3
    assert rec["invalid_breakdown"] == {"bad_json": 1, "unknown_tool": 1, "schema_violation": 1}


def test_api_tool_use_failed_counts_invalid_and_continues():
    c, _ = mk([
        FakeResp(400, {"error": {"code": "tool_use_failed", "message": "Failed to call a function"}}),
        completion("12"),
    ])
    rec = run_one(c, "m", TASKS_BY_ID["t01"], 1, "default", None)
    assert rec["success"] and rec["invalid_tool_calls"] == 1
    assert rec["invalid_breakdown"] == {"api_tool_use_failed": 1}


def test_failed_run_is_logged_not_dropped():
    c, _ = mk([FakeResp(400, {"error": {"code": "other", "message": "bad request"}})])
    rec = run_one(c, "m", TASKS_BY_ID["t01"], 1, "default", None)
    assert rec["error"] == "http_400" and rec["success"] is False and rec["stop_reason"] == "error"


def test_max_steps_stops_loop():
    items = [completion(tool_calls=[("list_items", {})]) for _ in range(3)]
    c, _ = mk(items)
    rec = run_one(c, "m", TASKS_BY_ID["t01"], 1, "default", None, max_steps=3)
    assert rec["steps"] == 3 and rec["stop_reason"] == "max_steps" and not rec["success"]


def test_quota_propagates_out_of_run_one():
    c, _ = mk([FakeResp(429, {"error": {"message": "requests per day (RPD) exceeded"}})])
    with pytest.raises(QuotaExhausted):
        run_one(c, "m", TASKS_BY_ID["t01"], 1, "default", None)
