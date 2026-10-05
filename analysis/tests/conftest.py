"""Test setup. Tests write only to pytest temp dirs; a session-level check
fails if anything under analysis/results changes while the tests run."""
import hashlib
import sys
from pathlib import Path

import pytest

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))


def _snapshot(d: Path) -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture(scope="session", autouse=True)
def committed_results_untouched():
    res = ANALYSIS / "results"
    before = _snapshot(res) if res.exists() else {}
    yield
    after = _snapshot(res) if res.exists() else {}
    assert before == after, "tests modified analysis/results"
