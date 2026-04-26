"""MCAP format reader for Forge.

This module currently provides scaffolding + format registration. Episode
yielding (driven by TopicConfig) lands in PR2.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from forge.core.exceptions import EpisodeNotFoundError, MissingDependencyError
from forge.core.models import DatasetInfo, Episode
from forge.formats.mcap.inspect import inspect_mcap
from forge.formats.mcap.topic_config import TopicConfig
from forge.formats.registry import FormatRegistry


def _check_mcap() -> None:
    try:
        import mcap  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            dependency="mcap",
            feature="MCAP format support",
            install_hint='pip install "forge-robotics[mcap]"',
        ) from e


@FormatRegistry.register_reader("mcap")
class MCAPReader:
    """Reader for MCAP files (ROS2 CDR + Foxglove Protobuf)."""

    @property
    def format_name(self) -> str:
        return "mcap"

    @classmethod
    def can_read(cls, path: Path) -> bool:
        if not path.exists():
            return False
        if path.is_file() and path.suffix.lower() == ".mcap":
            return True
        # Directory of split mcaps but NOT a ROS2 bag dir (those have metadata.yaml).
        if path.is_dir() and not (path / "metadata.yaml").exists():
            if any(path.glob("*.mcap")):
                return True
        return False

    @classmethod
    def detect_version(cls, path: Path) -> str | None:
        try:
            inv = inspect_mcap(path if path.is_file() else next(path.glob("*.mcap")))
        except Exception:
            return None
        return f"mcap/{inv.profile}" if inv.profile else "mcap"

    def inspect(self, path: Path) -> DatasetInfo:
        inv = inspect_mcap(path if path.is_file() else next(path.glob("*.mcap")))
        info = DatasetInfo(path=Path(path), format="mcap")
        info.format_version = f"mcap/{inv.profile}" if inv.profile else "mcap"
        info.num_episodes = 1  # Refined by reader once TopicConfig drives splitting.
        info.total_frames = inv.total_messages
        info.has_timestamps = inv.duration_ns is not None
        if inv.duration_ns and inv.total_messages:
            duration_sec = inv.duration_ns / 1e9
            if duration_sec > 0:
                info.inferred_fps = round(inv.total_messages / duration_sec, 2)
        info.metadata["mcap_profile"] = inv.profile
        info.metadata["mcap_channels"] = [
            {"topic": c.topic, "schema": c.schema_name, "messages": c.message_count}
            for c in inv.channels
        ]
        return info

    def read_episodes(
        self, path: Path, config: TopicConfig | None = None
    ) -> Iterator[Episode]:
        _check_mcap()
        raise NotImplementedError(
            "MCAPReader.read_episodes lands in PR2 (reader + sync). "
            "Use forge.formats.mcap.inspect_mcap() for channel/schema info."
        )

    def read_episode(self, path: Path, episode_id: str) -> Episode:
        for ep in self.read_episodes(path):
            if ep.episode_id == episode_id:
                return ep
        raise EpisodeNotFoundError(episode_id, path)
