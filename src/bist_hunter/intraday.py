"""Intraday HOLD/EXIT/CANCEL state machine for paper monitoring."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MonitorInput:
    entry_score: float
    current_score: float
    current_return: float
    stop_return: float = -0.03
    min_hold_score: float = 55.0


def decide(inp: MonitorInput) -> str:
    if inp.current_return <= inp.stop_return:
        return "EXIT"
    if inp.current_score < inp.min_hold_score and inp.current_score < inp.entry_score * 0.80:
        return "EXIT"
    if inp.current_score >= inp.entry_score * 0.95:
        return "HOLD"
    return "CANCEL"
