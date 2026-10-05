"""Twenty deterministic tasks. Each has a reference action sequence that defines the
expected end state (computed by replaying it on a fresh World) and, for question
tasks, an answer check. Success requires the full end state to match, so
unrequested side effects fail a task.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Callable

from world import World

SYSTEM_PROMPT = (
    "You are an assistant for a small order-management system. Use the provided tools "
    "to complete the user's request. Do not ask clarifying questions. When you are done, "
    "reply with a short final answer."
)


@dataclass
class Task:
    id: str
    prompt: str
    reference: list[tuple[str, dict]] = field(default_factory=list)
    answer_check: Callable[[str], bool] | None = None
    category: str = ""


def _word(pattern: str) -> Callable[[str], bool]:
    rx = re.compile(pattern, re.IGNORECASE)
    return lambda text: bool(rx.search(text or ""))


def _exact(expected: str) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        cleaned = re.sub(r"[^a-z]", "", (text or "").lower())
        return cleaned == expected

    return check


def _csv_names(expected: set[str]) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        parts = {re.sub(r"[^a-z]", "", p.lower()) for p in re.split(r"[,\s]+(?:and\s+)?", text or "")}
        parts.discard("")
        parts.discard("and")
        return parts == expected

    return check


TASKS: list[Task] = [
    Task("t01", "What is the unit price of a gadget? Reply with the number only.", [],
         _word(r"(?<![\d.])12(?![\d.])"), "lookup"),
    Task("t02", "How many sprockets are in stock? Reply with the number only.", [],
         _word(r"(?<![\d.])7(?![\d.])"), "lookup"),
    Task("t03", "Place an order for 3 widgets for alice.",
         [("place_order", {"customer": "alice", "item": "widget", "qty": 3})], None, "write"),
    Task("t05", "Order 4 gadgets for bob.", [], None, "refusal"),
    Task("t06", "Cancel order O-1002.", [("cancel_order", {"order_id": "O-1002"})], None, "write"),
    Task("t07", "Cancel order O-9999.", [], None, "refusal"),
    Task("t08", "Cancel all of alice's open orders.",
         [("cancel_order", {"order_id": "O-1001"}), ("cancel_order", {"order_id": "O-1003"})],
         None, "multi-step"),
    Task("t09", "What is the total cost of orders O-1001 and O-1003 combined, at current unit prices? "
                "Reply with the number only.", [], _word(r"(?<![\d.])37(?![\d.])"), "multi-step"),
    Task("t10", "Add 5 units to gizmo stock, then order 2 gizmos for carol.",
         [("restock", {"item": "gizmo", "qty": 5}),
          ("place_order", {"customer": "carol", "item": "gizmo", "qty": 2})], None, "multi-step"),
    Task("t11", "If doohickey stock is above 20, order 5 doohickeys for bob; otherwise do nothing.",
         [("place_order", {"customer": "bob", "item": "doohickey", "qty": 5})], None, "conditional"),
    Task("t12", "If gizmo stock is above 0, order 1 gizmo for alice; otherwise do nothing.",
         [], None, "conditional"),
    Task("t13", "Which of widget, gadget and sprocket has the lowest stock? "
                "Reply with only the item name.", [], _exact("gadget"), "lookup"),
    Task("t14", "What is the combined balance of alice and bob? Reply with the number only.", [],
         _word(r"(?<![\d.])140(?![\d.])"), "multi-step"),
    Task("t16", "Place two separate orders for bob: one for 1 widget and one for 2 sprockets.",
         [("place_order", {"customer": "bob", "item": "widget", "qty": 1}),
          ("place_order", {"customer": "bob", "item": "sprocket", "qty": 2})], None, "multi-step"),
    Task("t17", "Cancel carol's open order, then place an order for 4 gadgets for carol.",
         [("cancel_order", {"order_id": "O-1004"}),
          ("place_order", {"customer": "carol", "item": "gadget", "qty": 4})], None, "multi-step"),
    Task("t18", "Make sure gadget stock is at least 10 by adding only the missing units.",
         [("restock", {"item": "gadget", "qty": 6})], None, "conditional"),
    Task("t19", "Add a note for alice with the text: refund approved",
         [("add_note", {"customer": "alice", "text": "refund approved"})], None, "write"),
    Task("t20", "How many open orders does alice have? Reply with the number only.", [],
         _word(r"(?<![\d.])2(?![\d.])"), "lookup"),
    # t21 and t22 replace t04 and t15 (see README, "Task revision"): t04 did not forbid
    # restocking, and t15 could not be answered because no tool enumerates customers.
    Task("t21", "Order 2 gizmos for carol. Do not change stock levels.", [], None, "refusal"),
    Task("t22", "Is carol a gold tier customer? Reply yes or no.", [], _exact("yes"), "lookup"),
]

TASKS_BY_ID = {t.id: t for t in TASKS}


def expected_state(task: Task) -> dict:
    w = World()
    for name, args in task.reference:
        _, err = w.call(name, args)
        assert not err, f"reference action failed for {task.id}: {name} {args}"
    return w.state


def _norm(state: dict) -> dict:
    s = copy.deepcopy(state)
    s["notes"] = {
        k: [re.sub(r"[\s.]+$", "", t.strip().lower()) for t in v] for k, v in s["notes"].items()
    }
    return s


def check_success(task: Task, final_state: dict, final_answer: str | None) -> bool:
    if _norm(final_state) != _norm(expected_state(task)):
        return False
    if task.answer_check is not None and not task.answer_check(final_answer or ""):
        return False
    return True
