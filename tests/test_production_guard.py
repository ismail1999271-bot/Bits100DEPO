import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_no_order_routing.py"


def _load():
    spec = importlib.util.spec_from_file_location("guard", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_has_no_order_routing():
    assert _load().scan() == []


def test_guard_pattern_detects_order_calls():
    guard = _load()
    assert guard.FORBIDDEN.search("client.create_order(symbol='X')")
    assert guard.FORBIDDEN.search("api.cancel_order (1)")
    assert not guard.FORBIDDEN.search("raise ValueError('invalid loss/order limits')")
