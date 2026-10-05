import json


class FakeResp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def completion(content=None, tool_calls=None, finish="stop", pt=100, ct=10):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {"id": f"c{i}", "type": "function",
             "function": {"name": n, "arguments": a if isinstance(a, str) else json.dumps(a)}}
            for i, (n, a) in enumerate(tool_calls)
        ]
        finish = "tool_calls"
    return FakeResp(200, {"choices": [{"message": msg, "finish_reason": finish}],
                          "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}})


class Scripted:
    """Transport returning a scripted list of responses (or raising exceptions)."""

    def __init__(self, items):
        self.items = list(items)
        self.payloads = []

    def __call__(self, url, headers, payload, timeout):
        self.payloads.append(payload)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
