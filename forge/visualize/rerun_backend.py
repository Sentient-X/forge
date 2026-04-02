"""Rerun visualization backend for Forge.

Logs episodes from any Forge-supported format into the Rerun viewer.
Images are logged as rr.Image, per-dimension action/state as rr.Scalars,
and segment labels as rr.TextLog, all aligned on a shared "frame" timeline.

Strategy: write a .rrd recording file first, then open it with the rerun
CLI. This guarantees all data is flushed before the viewer opens it,
avoiding the race condition with spawn=True's gRPC server.

Usage:
    forge visualize ./dataset --backend rerun
    forge visualize ./dataset --backend rerun --episode 2 --segment
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

from forge.core.models import Episode
from forge.formats.registry import FormatRegistry


def _require_rerun() -> object:
    try:
        import rerun as rr
        return rr
    except ImportError:
        from forge.core.exceptions import MissingDependencyError
        raise MissingDependencyError(
            dependency="rerun-sdk",
            feature="Rerun visualization",
            install_hint="pip install rerun-sdk  # or: pip install forge-robotics[rerun]",
        )


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        return (img * 255).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def log_episode(
    rr: object,
    episode: Episode,
    segments: list[dict] | None = None,
    frame_offset: int = 0,
) -> int:
    """Log a single episode into the active Rerun recording.

    Each frame is logged under the "frame" sequence timeline so that images,
    scalars, and text logs are all temporally aligned in the viewer.

    Args:
        rr: The imported rerun module.
        episode: Forge Episode to visualize.
        segments: Optional list of segment dicts with keys "start", "end",
            "label" (as produced by the segmentation subsystem).
        frame_offset: Global frame counter offset so multiple episodes are
            placed sequentially on the same timeline without overlap.

    Returns:
        Total number of frames logged (for computing the next offset).
    """
    # Build a per-frame segment label lookup from start/end ranges.
    frame_labels: dict[int, str] = {}
    if segments:
        for seg in segments:
            label = seg.get("label") or f"segment_{seg['start']}"
            frame_labels[seg["start"]] = label

    frames = list(episode.frames())

    # Mark the episode boundary on the timeline.
    rr.set_time("frame", sequence=frame_offset)  # type: ignore[attr-defined]
    rr.log("episodes", rr.TextLog(f"episode {episode.episode_id}"))  # type: ignore[attr-defined]

    for i, frame in enumerate(frames):
        rr.set_time("frame", sequence=frame_offset + i)  # type: ignore[attr-defined]

        # ── Images ──────────────────────────────────────────────────────────
        for key, lazy_img in frame.images.items():
            img = _to_uint8(lazy_img.load())
            # Log all camera images under camera/<key>.  The "rgb" / "image"
            # hint the user mentioned is just a naming convention; since
            # frame.images only contains image data we log all of them.
            rr.log(f"camera/{key}", rr.Image(img))  # type: ignore[attr-defined]

        # ── Actions ─────────────────────────────────────────────────────────
        if frame.action is not None:
            action = np.asarray(frame.action).flatten()
            for dim, val in enumerate(action):
                rr.log(f"action/{dim}", rr.Scalars(float(val)))  # type: ignore[attr-defined]

        # ── Proprioception / state ───────────────────────────────────────────
        if frame.state is not None:
            state = np.asarray(frame.state).flatten()
            for dim, val in enumerate(state):
                rr.log(f"state/{dim}", rr.Scalars(float(val)))  # type: ignore[attr-defined]

        # ── Segment labels ──────────────────────────────────────────────────
        if i in frame_labels:
            rr.log("segments", rr.TextLog(frame_labels[i]))  # type: ignore[attr-defined]

    return len(frames)


def visualize_rerun(
    dataset_path: Path,
    episode_idx: int = 0,
    segment: bool = False,
    max_episodes: int = 1,
) -> None:
    """Open the Rerun viewer and stream episode data into it.

    Args:
        dataset_path: Path to any Forge-supported dataset.
        episode_idx: Which episode to visualize (0-based).
        segment: If True, run PELT segmentation and annotate the timeline.
        max_episodes: Number of episodes to log (default 1).
    """
    rr = _require_rerun()
    import rerun.blueprint as rrb

    rr.init("forge")  # type: ignore[attr-defined]

    format_name = FormatRegistry.detect_format(dataset_path)
    reader = FormatRegistry.get_reader(format_name)

    # Optional segmentation
    seg_results: dict[str, list[dict]] = {}
    if segment:
        try:
            from forge.segment.analyzer import SegmentAnalyzer
            from forge.segment.config import SegmentConfig
            from forge.segment.labeler import PhaseLabeler
        except ImportError:
            print("Segmentation requires ruptures: pip install forge-robotics[segment]")
            segment = False

    episodes_logged = 0
    blueprint_sent = False
    global_frame_offset = 0

    for ep_i, episode in enumerate(reader.read_episodes(dataset_path)):
        if ep_i < episode_idx:
            continue
        if episodes_logged >= max_episodes:
            break

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)  # type: ignore[attr-defined]

        # Materialise frames once so we can both inspect schema and log.
        first_frames = list(episode.frames())
        episode._frames_cache = first_frames  # noqa: SLF001
        if not first_frames:
            continue
        first_frame = first_frames[0]

        # Build and send the blueprint exactly once, before any data.
        if not blueprint_sent:
            # ── Left column: camera(s) stacked vertically ─────────────────
            camera_views = [
                rrb.Spatial2DView(name=key, origin=f"camera/{key}")
                for key in first_frame.images
            ]
            left = rrb.Vertical(*camera_views) if len(camera_views) > 1 else camera_views[0]

            # ── Right column: scalars on top, episode log pinned at bottom ─
            right_rows: list = []
            if first_frame.action is not None:
                right_rows.append(rrb.TimeSeriesView(name="action", origin="action"))
            if first_frame.state is not None:
                right_rows.append(rrb.TimeSeriesView(name="state", origin="state"))
            if segment:
                right_rows.append(rrb.TextLogView(name="segments", origin="segments"))
            right_rows.append(rrb.TextLogView(name="episodes", origin="episodes"))

            # Give scalars most of the vertical space; text log gets 12%.
            n_scalar = len(right_rows) - 1
            text_share = 0.12
            scalar_share = (1.0 - text_share) / n_scalar if n_scalar else 1.0
            row_shares = [scalar_share] * n_scalar + [text_share]
            right = rrb.Vertical(*right_rows, row_shares=row_shares)

            # ── Assemble: cameras 45 %, scalars 55 % ──────────────────────
            blueprint = rrb.Blueprint(
                rrb.Horizontal(left, right, column_shares=[0.45, 0.55]),
                collapse_panels=True,
            )
            rr.send_blueprint(blueprint)  # type: ignore[attr-defined]
            blueprint_sent = True

        segments: list[dict] | None = None
        if segment:
            frames = list(episode.frames())
            states = np.array([
                np.asarray(f.state).flatten()
                for f in frames
                if f.state is not None
            ])
            if states.ndim == 2 and len(states) > 4:
                config = SegmentConfig(label_phases=True, cost_model="l2")
                analyzer = SegmentAnalyzer(config=config)
                es = analyzer.segment_episode_arrays(
                    episode_id=episode.episode_id,
                    signal=states,
                )
                if config.label_phases and es.segments:
                    labeler = PhaseLabeler()
                    labeler.label_segments(es.segments, states)
                segments = [
                    {"start": s.start, "end": s.end, "label": s.label}
                    for s in es.segments
                ]
            # Re-use already-materialized frames via cached loader
            episode._frames_cache = frames  # noqa: SLF001

        print(f"Logging episode {episode.episode_id} ({ep_i}) to Rerun...")
        n_frames = log_episode(rr, episode, segments=segments, frame_offset=global_frame_offset)
        global_frame_offset += n_frames
        episodes_logged += 1

    # Save to ~/.cache/forge/recordings/ so recordings persist and can be
    # re-opened later with: rerun ~/.cache/forge/recordings/<file>.rrd
    recordings_dir = Path.home() / ".cache" / "forge" / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    rrd_path = recordings_dir / f"{dataset_path.name}_{timestamp}.rrd"

    print(f"Saving recording ({episodes_logged} episode(s)) → {rrd_path}")
    rr.save(str(rrd_path))  # type: ignore[attr-defined]
    print(f"Opening Rerun viewer... (re-open later with: rerun {rrd_path})")
    subprocess.Popen(["rerun", str(rrd_path)])
