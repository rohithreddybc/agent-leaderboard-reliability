"""Deterministic in-memory order-management world with nine small tools.

Everything is local and seeded: there is no randomness and no network. Each run
gets a fresh deep copy of INITIAL_STATE, so a task's end-state check depends only
on the tool calls made during that run.
"""
from __future__ import annotations

import copy
import json
from typing import Any

INITIAL_STATE: dict[str, Any] = {
    "stock": {"widget": 10, "gadget": 4, "gizmo": 0, "doohickey": 25, "sprocket": 7},
    "prices": {"widget": 5, "gadget": 12, "gizmo": 30, "doohickey": 2, "sprocket": 9},
    "customers": {
        "alice": {"balance": 100, "tier": "basic"},
        "bob": {"balance": 40, "tier": "basic"},
        "carol": {"balance": 250, "tier": "gold"},
    },
    "orders": {
        "O-1001": {"customer": "alice", "item": "widget", "qty": 2, "status": "open"},
        "O-1002": {"customer": "bob", "item": "gadget", "qty": 1, "status": "open"},
        "O-1003": {"customer": "alice", "item": "sprocket", "qty": 3, "status": "open"},
        "O-1004": {"customer": "carol", "item": "doohickey", "qty": 10, "status": "open"},
    },
    "notes": {},
    "next_order": 1005,
}


def _fn(name: str, description: str, props: dict[str, dict], required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


_S = {"type": "string"}
_I = {"type": "integer"}

TOOL_SCHEMAS: list[dict] = [
    _fn("list_items", "List all item names.", {}, []),
    _fn("get_item", "Get stock and unit price of an item.", {"item": _S}, ["item"]),
    _fn("get_customer", "Get a customer's balance and tier.", {"customer": _S}, ["customer"]),
    _fn("list_orders", "List order ids of a customer.", {"customer": _S}, ["customer"]),
    _fn("get_order", "Get an order by id.", {"order_id": _S}, ["order_id"]),
    _fn(
        "place_order",
        "Place an order. Debits the customer's balance and reduces stock.",
        {"customer": _S, "item": _S, "qty": _I},
        ["customer", "item", "qty"],
    ),
    _fn(
        "cancel_order",
        "Cancel an open order. Refunds the customer and returns stock.",
        {"order_id": _S},
        ["order_id"],
    ),
    _fn("restock", "Add units to an item's stock.", {"item": _S, "qty": _I}, ["item", "qty"]),
    _fn("add_note", "Attach a text note to a customer.", {"customer": _S, "text": _S}, ["customer", "text"]),
]

SCHEMA_BY_NAME = {t["function"]["name"]: t["function"]["parameters"] for t in TOOL_SCHEMAS}

# Invalid-call categories
UNKNOWN_TOOL = "unknown_tool"
BAD_JSON = "bad_json"
SCHEMA_VIOLATION = "schema_violation"


def validate_call(name: str, raw_args: Any) -> tuple[dict | None, str | None, str | None]:
    """Return (args, category, message). category is None when the call is valid."""
    if name not in SCHEMA_BY_NAME:
        return None, UNKNOWN_TOOL, f"unknown tool '{name}'"
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError as e:
            return None, BAD_JSON, f"arguments are not valid JSON ({e.msg})"
    else:
        args = raw_args
    if not isinstance(args, dict):
        return None, BAD_JSON, "arguments must be a JSON object"
    schema = SCHEMA_BY_NAME[name]
    props = schema["properties"]
    for key in schema["required"]:
        if key not in args:
            return None, SCHEMA_VIOLATION, f"missing required argument '{key}'"
    for key, value in args.items():
        if key not in props:
            return None, SCHEMA_VIOLATION, f"unexpected argument '{key}'"
        want = props[key]["type"]
        if want == "string" and not isinstance(value, str):
            return None, SCHEMA_VIOLATION, f"argument '{key}' must be a string"
        if want == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            return None, SCHEMA_VIOLATION, f"argument '{key}' must be an integer"
    return args, None, None


class World:
    def __init__(self) -> None:
        self.state = copy.deepcopy(INITIAL_STATE)

    # -- tool implementations: each returns a JSON-serialisable result or raises ToolError
    def call(self, name: str, args: dict) -> tuple[str, bool]:
        """Execute a validated call. Returns (result_text, is_tool_error)."""
        try:
            result = getattr(self, f"_t_{name}")(**args)
            return json.dumps(result), False
        except ToolError as e:
            return f"ERROR: {e}", True

    def _item(self, item: str) -> str:
        key = item.strip().lower()
        if key not in self.state["stock"]:
            raise ToolError(f"unknown item '{item}'")
        return key

    def _cust(self, customer: str) -> str:
        key = customer.strip().lower()
        if key not in self.state["customers"]:
            raise ToolError(f"unknown customer '{customer}'")
        return key

    def _t_list_items(self):
        return sorted(self.state["stock"])

    def _t_get_item(self, item):
        k = self._item(item)
        return {"item": k, "stock": self.state["stock"][k], "price": self.state["prices"][k]}

    def _t_get_customer(self, customer):
        k = self._cust(customer)
        return {"customer": k, **self.state["customers"][k]}

    def _t_list_orders(self, customer):
        k = self._cust(customer)
        return [oid for oid, o in self.state["orders"].items() if o["customer"] == k]

    def _t_get_order(self, order_id):
        o = self.state["orders"].get(order_id.strip())
        if o is None:
            raise ToolError(f"unknown order '{order_id}'")
        return {"order_id": order_id.strip(), **o}

    def _t_place_order(self, customer, item, qty):
        c, i = self._cust(customer), self._item(item)
        if qty < 1:
            raise ToolError("qty must be at least 1")
        if self.state["stock"][i] < qty:
            raise ToolError(f"insufficient stock for '{i}' (have {self.state['stock'][i]})")
        cost = qty * self.state["prices"][i]
        if self.state["customers"][c]["balance"] < cost:
            raise ToolError(f"insufficient balance (cost {cost}, have {self.state['customers'][c]['balance']})")
        self.state["stock"][i] -= qty
        self.state["customers"][c]["balance"] -= cost
        oid = f"O-{self.state['next_order']}"
        self.state["next_order"] += 1
        self.state["orders"][oid] = {"customer": c, "item": i, "qty": qty, "status": "open"}
        return {"order_id": oid, "cost": cost}

    def _t_cancel_order(self, order_id):
        o = self.state["orders"].get(order_id.strip())
        if o is None:
            raise ToolError(f"unknown order '{order_id}'")
        if o["status"] != "open":
            raise ToolError(f"order is not open (status {o['status']})")
        refund = o["qty"] * self.state["prices"][o["item"]]
        o["status"] = "cancelled"
        self.state["stock"][o["item"]] += o["qty"]
        self.state["customers"][o["customer"]]["balance"] += refund
        return {"order_id": order_id.strip(), "refund": refund}

    def _t_restock(self, item, qty):
        k = self._item(item)
        if qty < 1:
            raise ToolError("qty must be at least 1")
        self.state["stock"][k] += qty
        return {"item": k, "stock": self.state["stock"][k]}

    def _t_add_note(self, customer, text):
        k = self._cust(customer)
        self.state["notes"].setdefault(k, []).append(text)
        return {"customer": k, "notes": len(self.state["notes"][k])}


class ToolError(Exception):
    pass
