"""Unit tests for forge.formats.mcap.sync — pure functions, synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from forge.formats.mcap.sync import align, align_stream, hold, interpolate, nearest


@pytest.fixture
def numeric_stream() -> tuple[list[int], list[np.ndarray]]:
    ts = [0, 100, 200, 300, 400]  # ns spaced 100ns
    vals = [np.array([float(t)], dtype=np.float32) for t in ts]
    return ts, vals


# ---------------------------------------------------------------------------
# nearest
# ---------------------------------------------------------------------------


class TestNearest:
    def test_exact_match(self, numeric_stream):
        ts, vals = numeric_stream
        v, skew = nearest(200, ts, vals)
        assert v == vals[2]
        assert skew == 0

    def test_picks_left_on_tie(self, numeric_stream):
        ts, vals = numeric_stream
        v, skew = nearest(150, ts, vals)
        # Left (100) and right (200) both at distance 50 — implementation picks left.
        assert v == vals[1]
        assert skew == 50

    def test_clamps_to_first(self, numeric_stream):
        ts, vals = numeric_stream
        v, skew = nearest(-50, ts, vals)
        assert v == vals[0]
        assert skew == 50

    def test_clamps_to_last(self, numeric_stream):
        ts, vals = numeric_stream
        v, skew = nearest(500, ts, vals)
        assert v == vals[-1]
        assert skew == 100

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            nearest(0, [], [])


# ---------------------------------------------------------------------------
# hold (zero-order hold)
# ---------------------------------------------------------------------------


class TestHold:
    def test_returns_at_or_before(self, numeric_stream):
        ts, vals = numeric_stream
        v, skew = hold(250, ts, vals)
        assert v == vals[2]  # last <= 250 is index 2 (ts=200)
        assert skew == 50

    def test_exact_match_zero_skew(self, numeric_stream):
        ts, vals = numeric_stream
        v, skew = hold(300, ts, vals)
        assert v == vals[3]
        assert skew == 0

    def test_before_first_returns_first(self, numeric_stream):
        ts, vals = numeric_stream
        v, skew = hold(-50, ts, vals)
        assert v == vals[0]
        assert skew == 50  # positive: data hasn't started yet


# ---------------------------------------------------------------------------
# interpolate
# ---------------------------------------------------------------------------


class TestInterpolate:
    def test_linear_midpoint_numeric(self, numeric_stream):
        ts, vals = numeric_stream
        v, skew = interpolate(150, ts, vals)
        assert v == pytest.approx(np.array([150.0], dtype=np.float32))
        assert skew == 50

    def test_quarter_point(self):
        ts = [0, 1000]
        vals = [np.array([0.0]), np.array([4.0])]
        v, _ = interpolate(250, ts, vals)
        assert v == pytest.approx(np.array([1.0]))

    def test_falls_back_to_nearest_for_non_numeric(self):
        ts = [0, 100]
        vals = [{"k": "a"}, {"k": "b"}]
        v, _ = interpolate(40, ts, vals)
        assert v == {"k": "a"}

    def test_clamps_to_endpoints(self, numeric_stream):
        ts, vals = numeric_stream
        v_low, _ = interpolate(-100, ts, vals)
        v_high, _ = interpolate(900, ts, vals)
        assert v_low == vals[0]
        assert v_high == vals[-1]

    def test_preserves_dtype(self):
        ts = [0, 1000]
        vals = [np.array([0.0], dtype=np.float32), np.array([4.0], dtype=np.float32)]
        v, _ = interpolate(500, ts, vals)
        assert v.dtype == np.float32


# ---------------------------------------------------------------------------
# align dispatch + align_stream
# ---------------------------------------------------------------------------


def test_align_dispatch(numeric_stream):
    ts, vals = numeric_stream
    assert align(150, ts, vals, method="nearest")[0] == vals[1]
    assert align(150, ts, vals, method="hold")[0] == vals[1]
    assert align(150, ts, vals, method="interpolate")[0] == pytest.approx(np.array([150.0]))


def test_align_unknown_method_raises(numeric_stream):
    ts, vals = numeric_stream
    with pytest.raises(ValueError, match="unknown"):
        align(0, ts, vals, method="telepathy")  # type: ignore[arg-type]


def test_align_stream_returns_skews(numeric_stream):
    ts, vals = numeric_stream
    primary = [50, 150, 250]
    out_vals, skews = align_stream(primary, ts, vals, method="nearest")
    assert len(out_vals) == 3
    assert skews.tolist() == [50, 50, 50]
