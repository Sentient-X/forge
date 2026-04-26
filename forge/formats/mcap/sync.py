"""Multi-rate timestamp alignment for MCAP topics.

Pure functions, fully unit-testable with synthetic timestamp arrays.
The reader uses these to align secondary topic streams against a primary
topic's frame boundaries.

All timestamps are int nanoseconds. Numeric values are numpy arrays;
non-numeric values (raw bytes, strings, dicts) are passed through unchanged.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

SyncMethod = Literal["nearest", "interpolate", "hold"]


def _is_numeric(value: Any) -> bool:
    """True if value is a numeric numpy array suitable for interpolation."""
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    return False


def nearest(
    target_ns: int,
    src_ts_ns: list[int],
    src_values: list[Any],
) -> tuple[Any, int]:
    """Return (value, skew_ns) for the timestamp closest to target.

    Raises ValueError if src is empty.
    """
    if not src_ts_ns:
        raise ValueError("nearest: source stream is empty")
    idx = bisect_left(src_ts_ns, target_ns)
    if idx == 0:
        return src_values[0], abs(src_ts_ns[0] - target_ns)
    if idx == len(src_ts_ns):
        return src_values[-1], abs(src_ts_ns[-1] - target_ns)
    before, after = src_ts_ns[idx - 1], src_ts_ns[idx]
    if (target_ns - before) <= (after - target_ns):
        return src_values[idx - 1], target_ns - before
    return src_values[idx], after - target_ns


def hold(
    target_ns: int,
    src_ts_ns: list[int],
    src_values: list[Any],
) -> tuple[Any, int]:
    """Zero-order hold: most recent value at-or-before the target.

    If target precedes everything in the source, returns the first value
    with a positive skew (i.e. data hasn't started yet — we surface it
    rather than dropping the frame).
    """
    if not src_ts_ns:
        raise ValueError("hold: source stream is empty")
    idx = bisect_right(src_ts_ns, target_ns) - 1
    if idx < 0:
        return src_values[0], src_ts_ns[0] - target_ns
    return src_values[idx], target_ns - src_ts_ns[idx]


def interpolate(
    target_ns: int,
    src_ts_ns: list[int],
    src_values: list[Any],
) -> tuple[Any, int]:
    """Linear interpolation for numeric arrays; falls back to nearest otherwise.

    Returns (value, skew_ns). Skew is measured against the closer of the two
    bracket samples (so a target sitting exactly between two samples reports
    half the gap).
    """
    if not src_ts_ns:
        raise ValueError("interpolate: source stream is empty")
    idx = bisect_left(src_ts_ns, target_ns)
    if idx == 0:
        return src_values[0], src_ts_ns[0] - target_ns
    if idx == len(src_ts_ns):
        return src_values[-1], target_ns - src_ts_ns[-1]

    t_before, t_after = src_ts_ns[idx - 1], src_ts_ns[idx]
    v_before, v_after = src_values[idx - 1], src_values[idx]

    if not (_is_numeric(v_before) and _is_numeric(v_after)):
        return nearest(target_ns, src_ts_ns, src_values)

    span = t_after - t_before
    if span == 0:
        return v_before, 0
    alpha = (target_ns - t_before) / span
    interpolated = (1.0 - alpha) * np.asarray(v_before) + alpha * np.asarray(v_after)
    if isinstance(v_before, np.ndarray):
        interpolated = interpolated.astype(v_before.dtype, copy=False)
    skew = min(target_ns - t_before, t_after - target_ns)
    return interpolated, skew


_DISPATCH = {
    "nearest": nearest,
    "hold": hold,
    "interpolate": interpolate,
}


def align(
    target_ns: int,
    src_ts_ns: list[int],
    src_values: list[Any],
    method: SyncMethod = "nearest",
) -> tuple[Any, int]:
    """Dispatch to the named sync method. Returns (value, skew_ns)."""
    if method not in _DISPATCH:
        raise ValueError(f"unknown sync method: {method!r}")
    return _DISPATCH[method](target_ns, src_ts_ns, src_values)


def align_stream(
    primary_ts_ns: list[int],
    src_ts_ns: list[int],
    src_values: list[Any],
    method: SyncMethod = "nearest",
) -> tuple[list[Any], NDArray[np.int64]]:
    """Vectorize align() over every primary timestamp.

    Returns (values, skews_ns) — one entry per primary timestamp.
    """
    values: list[Any] = []
    skews: list[int] = []
    for t in primary_ts_ns:
        v, s = align(t, src_ts_ns, src_values, method)
        values.append(v)
        skews.append(s)
    return values, np.asarray(skews, dtype=np.int64)
