"""Tests for forge.formats.mcap.topic_config — load, validate, round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from forge.formats.mcap.topic_config import (
    ConfigValidationError,
    TopicConfig,
    dump_config,
    load_config,
    to_dict,
    validate_config,
)

VALID_YAML = """
source: ./teleop_session.mcap
episodes:
  strategy: marker
  marker_topic: /episode/start
  min_length_frames: 30
fields:
  observation.state:
    topic: /allegro/joint_states
    field: position
    dtype: float32
  action:
    topic: /allegro/commanded_position
    field: data
  observation.images.wrist:
    topic: /wrist_cam/image_raw/compressed
    encoding: jpeg
    target_shape: [240, 320, 3]
sync:
  primary: observation.state
  method: nearest
  max_skew_ms: 50
attachments:
  - name: robot_urdf
    path: ./allegro.urdf
    media_type: application/xml
task:
  description: "stack red on blue"
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(body)
    return p


def test_loads_full_valid_config(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, VALID_YAML))
    assert cfg.episodes.strategy == "marker"
    assert cfg.episodes.marker_topic == "/episode/start"
    assert cfg.episodes.min_length_frames == 30
    assert "observation.state" in cfg.fields
    assert cfg.fields["observation.state"].topic == "/allegro/joint_states"
    assert cfg.fields["observation.state"].field == "position"
    assert cfg.fields["observation.images.wrist"].target_shape == (240, 320, 3)
    assert cfg.sync.primary == "observation.state"
    assert cfg.sync.method == "nearest"
    assert cfg.sync.max_skew_ms == 50.0
    assert len(cfg.attachments) == 1 and cfg.attachments[0].name == "robot_urdf"
    assert cfg.task.description == "stack red on blue"


def test_resolves_source_relative_to_config(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, VALID_YAML))
    assert cfg.source is not None
    assert cfg.source.is_absolute()
    assert cfg.source.name == "teleop_session.mcap"


def test_episode_strategy_must_be_known(tmp_path: Path) -> None:
    body = "episodes:\n  strategy: bogus\nfields:\n  x:\n    topic: /a\n"
    with pytest.raises(ConfigValidationError, match="episodes.strategy"):
        load_config(_write(tmp_path, body))


def test_marker_strategy_requires_marker_topic(tmp_path: Path) -> None:
    body = "episodes:\n  strategy: marker\nfields:\n  x:\n    topic: /a\n"
    with pytest.raises(ConfigValidationError, match="marker_topic"):
        load_config(_write(tmp_path, body))


def test_time_gap_requires_seconds(tmp_path: Path) -> None:
    body = "episodes:\n  strategy: time_gap\nfields:\n  x:\n    topic: /a\n"
    with pytest.raises(ConfigValidationError, match="time_gap_seconds"):
        load_config(_write(tmp_path, body))


def test_segment_requires_signal(tmp_path: Path) -> None:
    body = "episodes:\n  strategy: segment\nfields:\n  x:\n    topic: /a\n"
    with pytest.raises(ConfigValidationError, match="segment_signal"):
        load_config(_write(tmp_path, body))


def test_field_requires_topic(tmp_path: Path) -> None:
    body = "fields:\n  observation.state:\n    field: position\n"
    with pytest.raises(ConfigValidationError, match="topic"):
        load_config(_write(tmp_path, body))


def test_sync_method_must_be_known(tmp_path: Path) -> None:
    body = "fields:\n  x:\n    topic: /a\nsync:\n  method: telepathy\n"
    with pytest.raises(ConfigValidationError, match="sync.method"):
        load_config(_write(tmp_path, body))


def test_sync_primary_must_reference_field(tmp_path: Path) -> None:
    body = "fields:\n  x:\n    topic: /a\nsync:\n  primary: nonexistent\n"
    with pytest.raises(ConfigValidationError, match="sync.primary"):
        load_config(_write(tmp_path, body))


def test_target_shape_must_be_list_of_ints(tmp_path: Path) -> None:
    body = (
        "fields:\n  obs:\n    topic: /a\n    target_shape: [240, '320', 3]\n"
    )
    with pytest.raises(ConfigValidationError, match="target_shape"):
        load_config(_write(tmp_path, body))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_round_trip_preserves_fields(tmp_path: Path) -> None:
    original = load_config(_write(tmp_path, VALID_YAML))
    out = tmp_path / "round.yaml"
    dump_config(original, out)
    reloaded = load_config(out)

    assert reloaded.episodes.strategy == original.episodes.strategy
    assert reloaded.episodes.marker_topic == original.episodes.marker_topic
    assert set(reloaded.fields) == set(original.fields)
    assert reloaded.fields["observation.state"].topic == original.fields["observation.state"].topic
    assert reloaded.fields["observation.images.wrist"].target_shape == (240, 320, 3)
    assert reloaded.sync.primary == original.sync.primary
    assert reloaded.task.description == original.task.description


def test_to_dict_is_yaml_safe(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, VALID_YAML))
    text = yaml.safe_dump(to_dict(cfg))
    assert "observation.state" in text
    assert yaml.safe_load(text) == to_dict(cfg)


def test_validate_empty_config_ok() -> None:
    cfg = TopicConfig()
    validate_config(cfg)
