"""OHLCV data-quality checks that fail closed on impossible market data."""
import pandas as pd


REQUIRED = ("symbol", "timestamp", "open", "high", "low", "close", "volume")


def validate_ohlcv(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        return [f"missing:{','.join(missing)}"]
    if frame[list(REQUIRED)].isna().any().any():
        errors.append("null_values")
    if (frame[["open", "high", "low", "close", "volume"]] < 0).any().any():
        errors.append("negative_values")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        errors.append("high_below_ohlc")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        errors.append("low_above_ohlc")
    if frame.duplicated(["symbol", "timestamp"]).any():
        errors.append("duplicate_bars")
    if not pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True)).is_monotonic_increasing:
        errors.append("timestamp_not_monotonic")
    return errors


def assert_ohlcv_quality(frame: pd.DataFrame) -> None:
    errors = validate_ohlcv(frame)
    if errors:
        raise ValueError("invalid OHLCV data: " + ", ".join(errors))
