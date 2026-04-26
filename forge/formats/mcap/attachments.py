"""MCAP attachment helpers — bundle on write, surface on read.

MCAP files can embed arbitrary file attachments (URDF, calibration JSON,
dataset stats). This module provides a thin functional interface so callers
don't need to drop down into mcap.records directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO

from forge.core.exceptions import MissingDependencyError


def _check_mcap() -> None:
    try:
        import mcap  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            dependency="mcap",
            feature="MCAP attachments",
            install_hint='pip install "forge-robotics[mcap]"',
        ) from e


@dataclass
class AttachmentRecord:
    """Materialized attachment as exposed by the reader."""

    name: str
    media_type: str
    data: bytes
    log_time_ns: int = 0
    create_time_ns: int = 0


def list_attachments(path: Path | str) -> list[AttachmentRecord]:
    """Return all attachments in an MCAP file (data fully loaded into memory)."""
    _check_mcap()
    from mcap.reader import make_reader

    out: list[AttachmentRecord] = []
    with Path(path).open("rb") as f:
        reader = make_reader(f)
        for att in reader.iter_attachments():
            out.append(
                AttachmentRecord(
                    name=att.name,
                    media_type=att.media_type,
                    data=att.data,
                    log_time_ns=att.log_time,
                    create_time_ns=att.create_time,
                )
            )
    return out


def extract_attachments(path: Path | str, dest_dir: Path | str) -> list[Path]:
    """Extract every attachment to dest_dir, preserving the attachment name.

    Returns the list of paths written. Existing files are overwritten.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for record in list_attachments(path):
        target = dest / record.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(record.data)
        written.append(target)
    return written


def add_attachment(
    writer_handle: IO,
    name: str,
    data: bytes,
    media_type: str,
    log_time_ns: int = 0,
    create_time_ns: int = 0,
) -> None:
    """Add an attachment via an open mcap.writer.Writer underlying writer.

    `writer_handle` is the low-level `mcap.writer.Writer` instance (the one
    held inside `mcap_ros2.writer.Writer._writer`). Most callers should use
    MCAPWriterConfig.attachments instead — this is the explicit API.
    """
    writer_handle.add_attachment(
        create_time=create_time_ns,
        log_time=log_time_ns,
        name=name,
        media_type=media_type,
        data=data,
    )
