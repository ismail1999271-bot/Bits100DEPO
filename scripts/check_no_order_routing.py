"""Production guard: the platform is research/paper-only and must never route orders.

Fails (exit 1) if any source file defines or calls exchange order-routing
functions, or if the paper broker cannot be imported.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "scripts", "dashboard")
FORBIDDEN = re.compile(
    r"\b(create_order|place_order|cancel_order|submit_order|new_order|send_order|"
    r"create_market_order|create_limit_order|order_market|order_limit)\s*\(",
    re.IGNORECASE,
)


def scan() -> list[str]:
    violations: list[str] = []
    for directory in SCAN_DIRS:
        for path in sorted((ROOT / directory).rglob("*.py")):
            if path.resolve() == Path(__file__).resolve():
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if FORBIDDEN.search(line):
                    violations.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    return violations


def main() -> int:
    from bist_hunter.paper_trading import PaperBroker  # noqa: F401

    violations = scan()
    if violations:
        print("ORDER_ROUTING_FORBIDDEN")
        print("\n".join(violations))
        return 1
    print("production guard OK: paper-only, no order routing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
