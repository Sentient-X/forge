"""Episode boundary computation from primary timestamps.

Implements the four strategies from TopicConfig.episodes:
  - single: one episode covering all primary timestamps
  - marker: split on each marker-topic message timestamp
  - time_gap: split when consecutive primary timestamps differ by > threshold
  - segment: PELT changepoint detection on a numeric signal stream

Returns a list of (start_index, end_index_exclusive) tuples into the primary
timestamp array.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Any

import numpy as np
from numpy.typing import NDArray

from forge.formats.mcap.topic_config import EpisodeSplit


def split_single(primary_ts_ns: list[int]) -> list[tuple[int, int]]:
    if not primary_ts_ns:
        return []
    return [(0, len(primary_ts_ns))]


def split_time_gap(
    primary_ts_ns: list[int], gap_seconds: float
) -> list[tuple[int, int]]:
    if not primary_ts_ns:
        return []
    gap_ns = int(gap_seconds * 1e9)
    boundaries: list[int] = [0]
    for i in range(1, len(primary_ts_ns)):
        if primary_ts_ns[i] - primary_ts_ns[i - 1] > gap_ns:
            boundaries.append(i)
    boundaries.append(len(primary_ts_ns))
    return list(zip(boundaries[:-1], boundaries[1:]))


def split_marker(
    primary_ts_ns: list[int], marker_ts_ns: list[int]
) -> list[tuple[int, int]]:
    """Each marker timestamp begins a new episode at the next primary frame.

    Markers before the first primary timestamp are treated as start-of-stream;
    markers after the last as end-of-stream.
    """
    if not primary_ts_ns:
        return []
    if not marker_ts_ns:
        return [(0, len(primary_ts_ns))]
    cuts = sorted({bisect_left(primary_ts_ns, t) for t in marker_ts_ns})
    cuts = [c for c in cuts if 0 < c < len(primary_ts_ns)]
    boundaries = [0] + cuts + [len(primary_ts_ns)]
    return list(zip(boundaries[:-1], boundaries[1:]))


def split_segment(
    signal: NDArray[np.floating[Any]], min_size: int = 30
) -> list[tuple[int, int]]:
    """PELT changepoint detection. Falls back to single segment if `ruptures`
    isn't installed."""
    if signal.size == 0:
        return []
    try:
        import ruptures as rpt
    except ImportError:
        return [(0, signal.shape[0])]

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
    cps = algo.predict(pen=10)  # ends with len(signal)
    boundaries = [0] + [c for c in cps if c < signal.shape[0]] + [signal.shape[0]]
    return list(zip(boundaries[:-1], boundaries[1:]))


def apply_post_filters(
    boundaries: list[tuple[int, int]],
    drop_first_n_frames: int,
    min_length_frames: int,
) -> list[tuple[int, int]]:
    """Drop leading frames + filter short episodes."""
    out: list[tuple[int, int]] = []
    for start, end in boundaries:
        s = start + drop_first_n_frames
        if s >= end:
            continue
        if (end - s) < min_length_frames:
            continue
        out.append((s, end))
    return out


def compute_boundaries(
    primary_ts_ns: list[int],
    spec: EpisodeSplit,
    marker_ts_ns: list[int] | None = None,
    segment_signal: NDArray[np.floating[Any]] | None = None,
) -> list[tuple[int, int]]:
    """Dispatch to the right strategy and apply post-filters."""
    if spec.strategy == "single":
        boundaries = split_single(primary_ts_ns)
    elif spec.strategy == "time_gap":
        if spec.time_gap_seconds is None:
            raise ValueError("time_gap strategy requires episodes.time_gap_seconds")
        boundaries = split_time_gap(primary_ts_ns, spec.time_gap_seconds)
    elif spec.strategy == "marker":
        boundaries = split_marker(primary_ts_ns, marker_ts_ns or [])
    elif spec.strategy == "segment":
        if segment_signal is None:
            raise ValueError("segment strategy requires a signal stream")
        boundaries = split_segment(segment_signal, min_size=max(spec.min_length_frames, 2))
    else:
        raise ValueError(f"unknown episode strategy: {spec.strategy!r}")
    return apply_post_filters(boundaries, spec.drop_first_n_frames, spec.min_length_frames)
