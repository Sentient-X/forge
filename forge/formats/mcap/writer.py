"""MCAP format writer for Forge.

Scaffolding only — the Episode -> MCAP encoder lands in PR3.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from forge.core.models import DatasetInfo, Episode
from forge.formats.registry import FormatRegistry


@dataclass
class MCAPWriterConfig:
    """Configuration for the MCAP writer."""

    chunk_compression: str = "zstd"  # "zstd" | "lz4" | "none"
    fps: float | None = None
    bundled_schemas_dir: Path | None = None
    attachments: list[Path] = field(default_factory=list)


@FormatRegistry.register_writer("mcap")
class MCAPWriter:
    @property
    def format_name(self) -> str:
        return "mcap"

    def write_episode(
        self,
        episode: Episode,
        output_path: Path,
        episode_index: int | None = None,
    ) -> None:
        raise NotImplementedError("MCAPWriter.write_episode lands in PR3.")

    def write_dataset(
        self,
        episodes: Iterator[Episode],
        output_path: Path,
        dataset_info: DatasetInfo | None = None,
    ) -> None:
        raise NotImplementedError("MCAPWriter.write_dataset lands in PR3.")

    def finalize(self, output_path: Path, dataset_info: DatasetInfo) -> None:
        raise NotImplementedError("MCAPWriter.finalize lands in PR3.")
