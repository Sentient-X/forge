"""Parametrized smoke tests over every Rerun MCAP fixture.

These fixtures span ROS2 CDR + Foxglove Protobuf + various message types.
At minimum, the inspector must read every file's channel list and schema set
without crashing — this is the floor for "first-class MCAP support".
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mcap")

from forge.formats import FormatRegistry  # noqa: E402
from forge.formats.mcap import MCAPReader, generate_config, inspect_mcap  # noqa: E402

FIXTURES = sorted(
    (Path(__file__).parent.parent / "sample_data" / "mcap").glob("*.mcap")
)

if not FIXTURES:
    pytest.skip("no MCAP fixtures present in sample_data/mcap/", allow_module_level=True)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.name)
def test_can_read_returns_true(fixture: Path) -> None:
    assert MCAPReader.can_read(fixture)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.name)
def test_format_registry_detects_mcap(fixture: Path) -> None:
    assert FormatRegistry.detect_format(fixture) == "mcap"


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.name)
def test_inspect_does_not_crash(fixture: Path) -> None:
    inv = inspect_mcap(fixture)
    assert inv.path == fixture
    assert len(inv.channels) > 0, "every fixture should have at least one channel"
    assert inv.total_messages > 0
    for ch in inv.channels:
        assert ch.topic
        # Schema name may legitimately be empty for raw byte channels;
        # we just require the field exists.
        assert hasattr(ch, "schema_name")


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.name)
def test_reader_inspect_returns_dataset_info(fixture: Path) -> None:
    info = MCAPReader().inspect(fixture)
    assert info.format == "mcap"
    assert info.total_frames > 0
    assert info.metadata["mcap_channels"], "channel inventory should be populated"


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.name)
def test_generate_config_produces_loadable_or_skips(fixture: Path) -> None:
    """For ROS-manipulation-shaped fixtures, generate_config should produce
    a non-empty TopicConfig; for non-mappable fixtures, it should skip with
    an informative reason rather than emit broken YAML."""
    res = generate_config(fixture)
    if res.skipped:
        assert res.reason, "skipped result must include a reason"
    else:
        assert res.config is not None
        assert len(res.config.fields) > 0
