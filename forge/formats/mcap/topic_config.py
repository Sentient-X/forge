"""Topic-to-Episode field mapping config for MCAP.

Defines a YAML-driven contract for: how to split an MCAP into episodes, how to
map MCAP topics + fields to Episode fields, multi-rate sync policy, and
attachments to bundle/extract.

Schema is intentionally a plain dataclass (not pydantic) since the project
doesn't pull in pydantic. Validation is explicit and raises ConfigValidationError
with the offending YAML field path in the message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from forge.core.exceptions import ForgeError


class ConfigValidationError(ForgeError):
    """Raised when a topic config fails validation."""


EpisodeStrategy = Literal["marker", "time_gap", "segment", "single"]
SyncMethod = Literal["nearest", "interpolate", "hold"]


@dataclass
class EpisodeSplit:
    """How to split an MCAP into episodes."""

    strategy: EpisodeStrategy = "single"
    marker_topic: str | None = None
    time_gap_seconds: float | None = None
    segment_signal: str | None = None
    min_length_frames: int = 0
    drop_first_n_frames: int = 0


@dataclass
class FieldMapping:
    """One MCAP-topic -> Episode-field mapping."""

    topic: str
    field: str | None = None
    dtype: str | None = None
    encoding: str | None = None
    target_shape: tuple[int, ...] | None = None
    # tf-specific
    frame: str | None = None
    parent: str | None = None


@dataclass
class SyncPolicy:
    primary: str | None = None
    method: SyncMethod = "nearest"
    max_skew_ms: float = 50.0


@dataclass
class Attachment:
    name: str
    path: Path | None = None
    media_type: str | None = None


@dataclass
class TaskSpec:
    description: str | None = None
    topic: str | None = None


@dataclass
class TopicConfig:
    """Full topic config — drives both reader and writer."""

    source: Path | None = None
    episodes: EpisodeSplit = field(default_factory=EpisodeSplit)
    fields: dict[str, FieldMapping] = field(default_factory=dict)
    sync: SyncPolicy = field(default_factory=SyncPolicy)
    attachments: list[Attachment] = field(default_factory=list)
    task: TaskSpec = field(default_factory=TaskSpec)


# ---------------------------------------------------------------------------
# Loading / validation
# ---------------------------------------------------------------------------


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigValidationError(msg)


def _coerce_field(name: str, raw: dict[str, Any]) -> FieldMapping:
    _require(isinstance(raw, dict), f"fields.{name}: must be a mapping")
    _require("topic" in raw, f"fields.{name}: 'topic' is required")
    target_shape = raw.get("target_shape")
    if target_shape is not None:
        _require(
            isinstance(target_shape, (list, tuple)) and all(isinstance(x, int) for x in target_shape),
            f"fields.{name}.target_shape: must be a list of ints",
        )
        target_shape = tuple(target_shape)
    return FieldMapping(
        topic=raw["topic"],
        field=raw.get("field"),
        dtype=raw.get("dtype"),
        encoding=raw.get("encoding"),
        target_shape=target_shape,
        frame=raw.get("frame"),
        parent=raw.get("parent"),
    )


def _coerce_episodes(raw: dict[str, Any] | None) -> EpisodeSplit:
    if raw is None:
        return EpisodeSplit()
    _require(isinstance(raw, dict), "episodes: must be a mapping")
    strategy = raw.get("strategy", "single")
    _require(
        strategy in ("marker", "time_gap", "segment", "single"),
        f"episodes.strategy: must be one of marker/time_gap/segment/single, got {strategy!r}",
    )
    if strategy == "marker":
        _require("marker_topic" in raw, "episodes.marker_topic: required when strategy=marker")
    if strategy == "time_gap":
        _require(
            "time_gap_seconds" in raw, "episodes.time_gap_seconds: required when strategy=time_gap"
        )
    if strategy == "segment":
        _require(
            "segment_signal" in raw, "episodes.segment_signal: required when strategy=segment"
        )
    return EpisodeSplit(
        strategy=strategy,
        marker_topic=raw.get("marker_topic"),
        time_gap_seconds=raw.get("time_gap_seconds"),
        segment_signal=raw.get("segment_signal"),
        min_length_frames=int(raw.get("min_length_frames", 0)),
        drop_first_n_frames=int(raw.get("drop_first_n_frames", 0)),
    )


def _coerce_sync(raw: dict[str, Any] | None) -> SyncPolicy:
    if raw is None:
        return SyncPolicy()
    _require(isinstance(raw, dict), "sync: must be a mapping")
    method = raw.get("method", "nearest")
    _require(
        method in ("nearest", "interpolate", "hold"),
        f"sync.method: must be one of nearest/interpolate/hold, got {method!r}",
    )
    return SyncPolicy(
        primary=raw.get("primary"),
        method=method,
        max_skew_ms=float(raw.get("max_skew_ms", 50.0)),
    )


def _coerce_attachments(raw: list[Any] | None) -> list[Attachment]:
    if raw is None:
        return []
    _require(isinstance(raw, list), "attachments: must be a list")
    out: list[Attachment] = []
    for i, item in enumerate(raw):
        _require(isinstance(item, dict), f"attachments[{i}]: must be a mapping")
        _require("name" in item, f"attachments[{i}].name: required")
        path = item.get("path")
        out.append(
            Attachment(
                name=item["name"],
                path=Path(path) if path else None,
                media_type=item.get("media_type"),
            )
        )
    return out


def _coerce_task(raw: dict[str, Any] | None) -> TaskSpec:
    if raw is None:
        return TaskSpec()
    _require(isinstance(raw, dict), "task: must be a mapping")
    return TaskSpec(description=raw.get("description"), topic=raw.get("topic"))


def validate_config(cfg: TopicConfig) -> None:
    """Cross-field validation. Raises ConfigValidationError on issues."""
    if cfg.sync.primary is not None:
        _require(
            cfg.sync.primary in cfg.fields,
            f"sync.primary: {cfg.sync.primary!r} not found in fields",
        )
    for name, mapping in cfg.fields.items():
        _require(bool(mapping.topic), f"fields.{name}.topic: must not be empty")


def load_config(path: Path | str) -> TopicConfig:
    """Load a YAML topic config from disk and validate it."""
    p = Path(path)
    _require(p.exists(), f"config file not found: {p}")
    raw = yaml.safe_load(p.read_text())
    return _from_dict(raw, base_dir=p.parent)


def _from_dict(raw: dict[str, Any] | None, base_dir: Path | None = None) -> TopicConfig:
    _require(raw is None or isinstance(raw, dict), "top-level config must be a mapping")
    raw = raw or {}

    source_raw = raw.get("source")
    source: Path | None = None
    if source_raw is not None:
        source = Path(source_raw)
        if base_dir is not None and not source.is_absolute():
            source = (base_dir / source).resolve()

    fields_raw = raw.get("fields") or {}
    _require(isinstance(fields_raw, dict), "fields: must be a mapping")
    fields = {name: _coerce_field(name, val) for name, val in fields_raw.items()}

    cfg = TopicConfig(
        source=source,
        episodes=_coerce_episodes(raw.get("episodes")),
        fields=fields,
        sync=_coerce_sync(raw.get("sync")),
        attachments=_coerce_attachments(raw.get("attachments")),
        task=_coerce_task(raw.get("task")),
    )
    validate_config(cfg)
    return cfg


def to_dict(cfg: TopicConfig) -> dict[str, Any]:
    """Serialize a TopicConfig back to a dict suitable for YAML dump."""
    out: dict[str, Any] = {}
    if cfg.source is not None:
        out["source"] = str(cfg.source)
    ep = cfg.episodes
    ep_dict = {"strategy": ep.strategy}
    if ep.marker_topic:
        ep_dict["marker_topic"] = ep.marker_topic
    if ep.time_gap_seconds is not None:
        ep_dict["time_gap_seconds"] = ep.time_gap_seconds
    if ep.segment_signal:
        ep_dict["segment_signal"] = ep.segment_signal
    if ep.min_length_frames:
        ep_dict["min_length_frames"] = ep.min_length_frames
    if ep.drop_first_n_frames:
        ep_dict["drop_first_n_frames"] = ep.drop_first_n_frames
    out["episodes"] = ep_dict

    fields_out: dict[str, dict[str, Any]] = {}
    for name, m in cfg.fields.items():
        d: dict[str, Any] = {"topic": m.topic}
        if m.field:
            d["field"] = m.field
        if m.dtype:
            d["dtype"] = m.dtype
        if m.encoding:
            d["encoding"] = m.encoding
        if m.target_shape:
            d["target_shape"] = list(m.target_shape)
        if m.frame:
            d["frame"] = m.frame
        if m.parent:
            d["parent"] = m.parent
        fields_out[name] = d
    out["fields"] = fields_out

    s = cfg.sync
    out["sync"] = {
        "primary": s.primary,
        "method": s.method,
        "max_skew_ms": s.max_skew_ms,
    }
    if cfg.attachments:
        out["attachments"] = [
            {
                "name": a.name,
                **({"path": str(a.path)} if a.path else {}),
                **({"media_type": a.media_type} if a.media_type else {}),
            }
            for a in cfg.attachments
        ]
    if cfg.task.description or cfg.task.topic:
        td: dict[str, Any] = {}
        if cfg.task.description:
            td["description"] = cfg.task.description
        if cfg.task.topic:
            td["topic"] = cfg.task.topic
        out["task"] = td
    return out


def dump_config(cfg: TopicConfig, path: Path | str) -> None:
    """Write a TopicConfig to a YAML file."""
    Path(path).write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
