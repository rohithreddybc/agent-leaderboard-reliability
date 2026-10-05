import pytest

from tasks import TASKS, check_success, expected_state
from world import INITIAL_STATE, World, validate_call, UNKNOWN_TOOL, BAD_JSON, SCHEMA_VIOLATION


def test_validate_call_categories():
    assert validate_call("nope", "{}")[1] == UNKNOWN_TOOL
    assert validate_call("get_item", "{not json")[1] == BAD_JSON
    assert validate_call("get_item", "[1]")[1] == BAD_JSON
    assert validate_call("get_item", "{}")[1] == SCHEMA_VIOLATION
    assert validate_call("get_item", '{"item":"x","extra":1}')[1] == SCHEMA_VIOLATION
    assert validate_call("place_order", '{"customer":"a","item":"b","qty":"3"}')[1] == SCHEMA_VIOLATION
    assert validate_call("place_order", '{"customer":"a","item":"b","qty":true}')[1] == SCHEMA_VIOLATION
    assert validate_call("list_items", "")[1] is None
    args, cat, _ = validate_call("get_item", '{"item":"widget"}')
    assert cat is None and args == {"item": "widget"}


def test_world_does_not_mutate_initial_state():
    w = World()
    w.call("place_order", {"customer": "alice", "item": "widget", "qty": 1})
    assert INITIAL_STATE["stock"]["widget"] == 10
    assert World().state == INITIAL_STATE


def test_failed_tool_calls_do_not_mutate():
    w = World()
    for name, args in [("place_order", {"customer": "bob", "item": "gadget", "qty": 4}),
                       ("place_order", {"customer": "carol", "item": "gizmo", "qty": 2}),
                       ("cancel_order", {"order_id": "O-9999"})]:
        _, err = w.call(name, args)
        assert err
    assert w.state == INITIAL_STATE


def test_twenty_unique_tasks():
    assert len(TASKS) == 20
    assert len({t.id for t in TASKS}) == 20


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_reference_solution_passes(task):
    answers = {"t01": "12", "t02": "7", "t09": "37", "t13": "gadget", "t14": "140",
               "t22": "Yes", "t20": "2"}
    assert check_success(task, expected_state(task), answers.get(task.id, "done"))


@pytest.mark.parametrize("task", [t for t in TASKS if t.reference], ids=lambda t: t.id)
def test_doing_nothing_fails_write_tasks(task):
    assert not check_success(task, World().state, "done")


def test_answer_checks_reject_wrong_answers():
    by = {t.id: t for t in TASKS}
    assert not check_success(by["t01"], World().state, "13")
    assert not check_success(by["t01"], World().state, "112")
    assert not check_success(by["t13"], World().state, "widget")
    assert not check_success(by["t22"], World().state, "no")
    assert check_success(by["t22"], World().state, "Yes.")
    assert not check_success(by["t09"], World().state, "")


def test_unrequested_side_effect_fails_a_lookup_task():
    by = {t.id: t for t in TASKS}
    w = World()
    w.call("restock", {"item": "widget", "qty": 1})
    assert not check_success(by["t01"], w.state, "12")


def test_note_normalisation():
    by = {t.id: t for t in TASKS}
    w = World()
    w.call("add_note", {"customer": "alice", "text": "Refund approved."})
    assert check_success(by["t19"], w.state, "ok")
