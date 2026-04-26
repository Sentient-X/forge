"""Attachment round-trip: write -> list -> extract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcap")
pytest.importorskip("mcap_ros2")

from forge.core.models import Episode, Frame  # noqa: E402
from forge.formats.mcap import (  # noqa: E402
    MCAPWriter,
    MCAPWriterConfig,
    extract_attachments,
    list_attachments,
)


def _trivial_episode() -> Episode:
    def gen():
        yield Frame(
            index=0,
            timestamp=0.0,
            state=np.zeros(3, dtype=np.float32),
            is_first=True,
            is_last=True,
        )

    return Episode(episode_id="ep0", _frame_loader=gen)


def test_write_and_list_attachment(tmp_path: Path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot name='dummy'/>")

    out = tmp_path / "with_att.mcap"
    MCAPWriter(MCAPWriterConfig(attachments=[urdf_path])).write_episode(
        _trivial_episode(), out
    )

    records = list_attachments(out)
    assert len(records) == 1
    rec = records[0]
    assert rec.name == "robot.urdf"
    assert rec.media_type == "application/xml"
    assert b"<robot" in rec.data


def test_extract_attachments_to_disk(tmp_path: Path):
    json_path = tmp_path / "calib.json"
    json_path.write_text('{"focal": 800}')
    out = tmp_path / "with_att.mcap"
    MCAPWriter(MCAPWriterConfig(attachments=[json_path])).write_episode(
        _trivial_episode(), out
    )

    dest = tmp_path / "extracted"
    written = extract_attachments(out, dest)
    assert len(written) == 1
    assert written[0].name == "calib.json"
    assert (dest / "calib.json").read_text() == '{"focal": 800}'


def test_missing_attachment_raises(tmp_path: Path):
    from forge.core.exceptions import ConversionError

    with pytest.raises(ConversionError, match="not found"):
        MCAPWriter(
            MCAPWriterConfig(attachments=[tmp_path / "nope.urdf"])
        ).write_episode(_trivial_episode(), tmp_path / "x.mcap")


def test_no_attachments_when_empty(tmp_path: Path):
    out = tmp_path / "no_att.mcap"
    MCAPWriter().write_episode(_trivial_episode(), out)
    assert list_attachments(out) == []
