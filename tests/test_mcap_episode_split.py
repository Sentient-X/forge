"""Unit tests for episode_split — synthetic timestamp arrays."""

from __future__ import annotations

import numpy as np
import pytest

from forge.formats.mcap import episode_split
from forge.formats.mcap.topic_config import EpisodeSplit


class TestSingle:
    def test_basic(self):
        assert episode_split.split_single([0, 100, 200]) == [(0, 3)]

    def test_empty(self):
        assert episode_split.split_single([]) == []


class TestTimeGap:
    def test_no_gap_one_episode(self):
        ts = [0, 100, 200, 300]
        assert episode_split.split_time_gap(ts, gap_seconds=1.0) == [(0, 4)]

    def test_one_gap_two_episodes(self):
        # gap of 2 seconds between idx 1 and 2
        ts = [0, 100, int(2.5e9), int(2.5e9) + 100]
        assert episode_split.split_time_gap(ts, gap_seconds=2.0) == [(0, 2), (2, 4)]

    def test_multiple_gaps(self):
        s = int(1.0e9)
        ts = [0, s + 1, s + 2, 3 * s + 3, 3 * s + 4]
        # gaps > 0.5s after idx 0 and idx 2
        assert episode_split.split_time_gap(ts, gap_seconds=0.5) == [(0, 1), (1, 3), (3, 5)]


class TestMarker:
    def test_no_markers_one_episode(self):
        ts = [0, 100, 200]
        assert episode_split.split_marker(ts, []) == [(0, 3)]

    def test_marker_inside_splits(self):
        ts = [0, 100, 200, 300, 400]
        # marker at 250 -> next primary >= 250 is index 3
        assert episode_split.split_marker(ts, [250]) == [(0, 3), (3, 5)]

    def test_marker_dedup(self):
        ts = [0, 100, 200, 300]
        assert episode_split.split_marker(ts, [110, 120, 150]) == [(0, 2), (2, 4)]

    def test_marker_outside_ignored(self):
        ts = [0, 100, 200]
        assert episode_split.split_marker(ts, [-100, 999]) == [(0, 3)]


class TestSegment:
    def test_falls_back_when_ruptures_missing(self, monkeypatch):
        # Force ImportError for ruptures.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ruptures":
                raise ImportError("forced")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        sig = np.array([1.0, 2.0, 3.0, 4.0])
        assert episode_split.split_segment(sig) == [(0, 4)]

    def test_empty_signal(self):
        assert episode_split.split_segment(np.array([])) == []


class TestPostFilters:
    def test_drop_first_n(self):
        bs = [(0, 100)]
        assert episode_split.apply_post_filters(bs, drop_first_n_frames=10, min_length_frames=0) == [(10, 100)]

    def test_min_length_filters_short(self):
        bs = [(0, 5), (10, 100)]
        assert episode_split.apply_post_filters(bs, drop_first_n_frames=0, min_length_frames=10) == [(10, 100)]

    def test_drop_consumes_entire_segment(self):
        bs = [(0, 5)]
        assert episode_split.apply_post_filters(bs, drop_first_n_frames=10, min_length_frames=0) == []


class TestComputeBoundaries:
    def test_dispatches_single(self):
        spec = EpisodeSplit(strategy="single")
        assert episode_split.compute_boundaries([0, 1, 2], spec) == [(0, 3)]

    def test_time_gap_requires_seconds(self):
        spec = EpisodeSplit(strategy="time_gap", time_gap_seconds=None)
        with pytest.raises(ValueError, match="time_gap_seconds"):
            episode_split.compute_boundaries([0, 1, 2], spec)

    def test_segment_requires_signal(self):
        spec = EpisodeSplit(strategy="segment")
        with pytest.raises(ValueError, match="signal"):
            episode_split.compute_boundaries([0, 1, 2], spec)

    def test_marker_no_marker_topic(self):
        spec = EpisodeSplit(strategy="marker")
        # No markers passed in -> single episode.
        assert episode_split.compute_boundaries([0, 1, 2], spec, marker_ts_ns=[]) == [(0, 3)]

    def test_post_filters_applied(self):
        spec = EpisodeSplit(strategy="single", drop_first_n_frames=5, min_length_frames=10)
        # Length after drop = 95, passes.
        assert episode_split.compute_boundaries(list(range(100)), spec) == [(5, 100)]
        # min_length 200 filters out.
        spec2 = EpisodeSplit(strategy="single", min_length_frames=200)
        assert episode_split.compute_boundaries(list(range(100)), spec2) == []
