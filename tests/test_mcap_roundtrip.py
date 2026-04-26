"""Round-trip tests for MCAPWriter -> MCAPReader.

Two flavors:
  - Synthetic Episode -> write -> read: assert exact equality on state/action
    and shape preservation on images. The synthetic flow exercises the writer
    in isolation from any source-format peculiarities.
  - Real Rerun fixture (Trossen) -> Forge MCAP -> re-read: assert that the
    mappable subset (state + action) survives the round trip within float
    tolerance. Images are encoded as raw rgb8 on the way out, so byte-equality
    isn't expected for h264-input fixtures, but shape and content should hold.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcap")
pytest.importorskip("mcap_ros2")

from forge.core.models import CameraInfo, Episode, Frame, LazyImage  # noqa: E402
from forge.formats.mcap import MCAPReader, MCAPWriter  # noqa: E402
from forge.formats.mcap.topic_config import (  # noqa: E402
    EpisodeSplit,
    FieldMapping,
    SyncPolicy,
    TopicConfig,
)
from forge.formats.mcap.writer import MCAPWriterConfig  # noqa: E402

FIXTURES = Path(__file__).parent.parent / "sample_data" / "mcap"
TROSSEN = FIXTURES / "trossen_transfer_cube.mcap"


@pytest.fixture(autouse=True)
def _silence_sync_warnings():
    logging.getLogger("forge.formats.mcap.reader").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Synthetic Episode helpers
# ---------------------------------------------------------------------------


def _synth_episode(
    n_frames: int = 20,
    state_dim: int = 7,
    action_dim: int = 7,
    img_hw: tuple[int, int] = (8, 8),
) -> Episode:
    rng = np.random.default_rng(seed=0)
    state_seq = rng.standard_normal((n_frames, state_dim)).astype(np.float32)
    action_seq = rng.standard_normal((n_frames, action_dim)).astype(np.float32)
    img_seq = rng.integers(0, 255, (n_frames, *img_hw, 3), dtype=np.uint8)

    def gen_frames():
        for i in range(n_frames):
            yield Frame(
                index=i,
                timestamp=float(i) * 0.05,  # 20Hz
                state=state_seq[i],
                action=action_seq[i],
                images={
                    "cam0": LazyImage(
                        loader=lambda im=img_seq[i]: im,
                        height=img_hw[0],
                        width=img_hw[1],
                        channels=3,
                    )
                },
                is_first=(i == 0),
                is_last=(i == n_frames - 1),
            )

    return Episode(
        episode_id="ep0",
        cameras={"cam0": CameraInfo(name="cam0", height=img_hw[0], width=img_hw[1], channels=3)},
        state_dim=state_dim,
        action_dim=action_dim,
        fps=20.0,
        _frame_loader=gen_frames,
    )


def _read_back(path: Path, *, n_cams: int = 1) -> Episode:
    """Read an MCAP back using the explicit topic layout the writer emits."""
    fields = {
        "observation.state": FieldMapping(
            topic="/forge/observation/state/ep_0000", field="position", dtype="float32"
        ),
        "action": FieldMapping(
            topic="/forge/action/ep_0000", field="position", dtype="float32"
        ),
    }
    for i in range(n_cams):
        cam_name = "cam0" if n_cams == 1 else f"cam{i}"
        fields[f"observation.images.{cam_name}"] = FieldMapping(
            topic=f"/forge/observation/images/{cam_name}/ep_0000"
        )
    cfg = TopicConfig(
        episodes=EpisodeSplit(strategy="single"),
        fields=fields,
        sync=SyncPolicy(primary="observation.state", method="nearest", max_skew_ms=1.0),
    )
    return next(MCAPReader().read_episodes(path, config=cfg))


# ---------------------------------------------------------------------------
# Synthetic round-trip
# ---------------------------------------------------------------------------


class TestSyntheticRoundTrip:
    def test_state_preserved_exactly(self, tmp_path: Path):
        original = _synth_episode()
        original_frames = list(original.frames())

        out = tmp_path / "synth.mcap"
        MCAPWriter().write_episode(original, out)
        readback = _read_back(out)
        readback_frames = readback.load_frames()

        assert len(readback_frames) == len(original_frames)
        for orig, back in zip(original_frames, readback_frames):
            np.testing.assert_array_equal(orig.state, back.state)

    def test_action_preserved_exactly(self, tmp_path: Path):
        original = _synth_episode()
        original_frames = list(original.frames())
        out = tmp_path / "synth.mcap"
        MCAPWriter().write_episode(original, out)
        readback = _read_back(out).load_frames()

        for orig, back in zip(original_frames, readback):
            np.testing.assert_array_equal(orig.action, back.action)

    def test_image_bytes_preserved(self, tmp_path: Path):
        """Raw rgb8 image bytes survive write -> read losslessly."""
        original = _synth_episode(n_frames=5, img_hw=(8, 8))
        original_frames = list(original.frames())
        out = tmp_path / "synth.mcap"
        MCAPWriter().write_episode(original, out)
        readback = _read_back(out).load_frames()

        for orig, back in zip(original_frames, readback):
            orig_img = orig.images["cam0"].load()
            back_img = back.images["cam0"].load()
            assert back_img.shape == orig_img.shape
            np.testing.assert_array_equal(back_img, orig_img)

    def test_timestamps_preserved(self, tmp_path: Path):
        original = _synth_episode(n_frames=5)
        original_frames = list(original.frames())
        out = tmp_path / "synth.mcap"
        MCAPWriter().write_episode(original, out)
        readback = _read_back(out).load_frames()

        for orig, back in zip(original_frames, readback):
            assert back.timestamp == pytest.approx(orig.timestamp, abs=1e-6)

    def test_frame_count_and_fps(self, tmp_path: Path):
        original = _synth_episode(n_frames=20)
        out = tmp_path / "synth.mcap"
        MCAPWriter().write_episode(original, out)
        readback = _read_back(out)

        frames = readback.load_frames()
        assert len(frames) == 20
        # 20Hz from synth -> primary timestamps span 0.0..0.95s -> ~20 fps
        assert readback.fps == pytest.approx(20.0, abs=0.5)

    def test_compression_zstd_default(self, tmp_path: Path):
        """Output file should actually open as a valid MCAP."""
        from mcap.reader import make_reader

        out = tmp_path / "synth.mcap"
        MCAPWriter().write_episode(_synth_episode(n_frames=5), out)
        with out.open("rb") as f:
            r = make_reader(f)
            summary = r.get_summary()
            assert summary is not None
            assert summary.statistics.message_count > 0

    def test_no_compression_option(self, tmp_path: Path):
        out = tmp_path / "synth_nc.mcap"
        MCAPWriter(MCAPWriterConfig(chunk_compression="none")).write_episode(
            _synth_episode(n_frames=3), out
        )
        assert out.exists() and out.stat().st_size > 0

    def test_invalid_compression_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="chunk_compression"):
            MCAPWriter(MCAPWriterConfig(chunk_compression="bogus")).write_episode(
                _synth_episode(n_frames=2), tmp_path / "x.mcap"
            )

    def test_writer_registered_in_format_registry(self):
        from forge.formats import FormatRegistry

        assert FormatRegistry.has_writer("mcap")
        w = FormatRegistry.get_writer("mcap")
        assert isinstance(w, MCAPWriter)

    def test_state_dim_inferred_from_readback(self, tmp_path: Path):
        out = tmp_path / "x.mcap"
        MCAPWriter().write_episode(_synth_episode(n_frames=5, state_dim=12), out)
        readback = _read_back(out)
        assert readback.state_dim == 12


# ---------------------------------------------------------------------------
# Real-fixture round-trip — Trossen
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TROSSEN.exists(), reason="trossen fixture missing")
class TestTrossenRoundTrip:
    """Read Trossen -> Forge MCAP -> re-read; assert state/action survive."""

    def test_state_action_survive(self, tmp_path: Path):
        # Read with hand-tuned config so the state/action mapping is unambiguous.
        config = TopicConfig(
            source=TROSSEN,
            episodes=EpisodeSplit(strategy="single"),
            fields={
                "observation.state": FieldMapping(
                    topic="/robot_left/joint_states", field="position", dtype="float32"
                ),
                "action": FieldMapping(
                    topic="/robot_right/joint_states", field="position", dtype="float32"
                ),
            },
            sync=SyncPolicy(
                primary="observation.state", method="nearest", max_skew_ms=200.0
            ),
        )
        original = next(MCAPReader().read_episodes(TROSSEN, config=config))
        original_frames = list(original.frames())
        assert len(original_frames) > 100

        # Write back as Forge MCAP.
        out = tmp_path / "trossen_roundtrip.mcap"
        MCAPWriter().write_episode(original, out)

        # Re-read and compare element-wise.
        readback = _read_back(out, n_cams=0)
        readback_frames = readback.load_frames()

        assert len(readback_frames) == len(original_frames)
        # Spot-check a handful of frames at indices 0, mid, last.
        for i in (0, len(original_frames) // 2, len(original_frames) - 1):
            np.testing.assert_allclose(
                original_frames[i].state,
                readback_frames[i].state,
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                original_frames[i].action,
                readback_frames[i].action,
                rtol=1e-5,
                atol=1e-5,
            )
