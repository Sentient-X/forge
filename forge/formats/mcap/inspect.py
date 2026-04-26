"""MCAP inspection + heuristic config generation.

Encoding-agnostic: walks the MCAP summary section via the `mcap` library,
listing channels, schemas, attachments, message stats. `generate_config()`
applies the heuristics from the design doc to propose a starter TopicConfig
(or skips with a reason if the file uses non-ROS schemas we can't map yet).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from forge.core.exceptions import InspectionError, MissingDependencyError
from forge.formats.mcap.topic_config import (
    EpisodeSplit,
    FieldMapping,
    SyncPolicy,
    TopicConfig,
)


def _check_mcap() -> None:
    try:
        import mcap  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            dependency="mcap",
            feature="MCAP format support",
            install_hint='pip install "forge-robotics[mcap]"',
        ) from e


# ---------------------------------------------------------------------------
# Inventory dataclasses (encoding-agnostic shape returned by inspect_mcap)
# ---------------------------------------------------------------------------


@dataclass
class ChannelInfo:
    topic: str
    schema_name: str
    schema_encoding: str  # "ros2msg", "protobuf", "jsonschema", ""
    message_encoding: str  # "cdr", "protobuf", "json", ""
    message_count: int = 0


@dataclass
class AttachmentInfo:
    name: str
    media_type: str
    size: int


@dataclass
class MCAPInventory:
    path: Path
    profile: str  # "ros2", "" (foxglove uses no profile), or other
    channels: list[ChannelInfo] = field(default_factory=list)
    attachments: list[AttachmentInfo] = field(default_factory=list)
    total_messages: int = 0
    compression: list[str] = field(default_factory=list)
    has_summary: bool = False
    duration_ns: int | None = None
    start_time_ns: int | None = None
    end_time_ns: int | None = None


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def inspect_mcap(path: Path | str) -> MCAPInventory:
    """Walk an MCAP file's summary section (or stream if no summary).

    Returns a structural inventory — no message payloads decoded.
    """
    _check_mcap()
    from mcap.reader import make_reader

    p = Path(path)
    if not p.exists():
        raise InspectionError(p, "file does not exist")

    inv = MCAPInventory(path=p, profile="")

    try:
        with p.open("rb") as f:
            reader = make_reader(f)
            header = reader.get_header()
            inv.profile = header.profile or ""

            summary = reader.get_summary()
            chan_msg_counts: dict[int, int] = {}

            if summary is not None:
                inv.has_summary = True
                stats = summary.statistics
                if stats is not None:
                    chan_msg_counts = dict(stats.channel_message_counts)
                    inv.total_messages = stats.message_count
                    if stats.message_start_time and stats.message_end_time:
                        inv.start_time_ns = stats.message_start_time
                        inv.end_time_ns = stats.message_end_time
                        inv.duration_ns = stats.message_end_time - stats.message_start_time

                comp_set: set[str] = set()
                for ci in summary.chunk_indexes:
                    comp_set.add(ci.compression or "none")
                inv.compression = sorted(comp_set)

                for chan_id, chan in summary.channels.items():
                    schema = summary.schemas.get(chan.schema_id)
                    inv.channels.append(
                        ChannelInfo(
                            topic=chan.topic,
                            schema_name=schema.name if schema else "",
                            schema_encoding=schema.encoding if schema else "",
                            message_encoding=chan.message_encoding,
                            message_count=chan_msg_counts.get(chan_id, 0),
                        )
                    )

                for att in summary.attachment_indexes:
                    inv.attachments.append(
                        AttachmentInfo(
                            name=att.name,
                            media_type=att.media_type,
                            size=att.data_size,
                        )
                    )
            else:
                # No summary — stream once to discover channels.
                seen: dict[int, ChannelInfo] = {}
                count = 0
                for schema, channel, msg in reader.iter_messages():
                    count += 1
                    if channel.id not in seen:
                        seen[channel.id] = ChannelInfo(
                            topic=channel.topic,
                            schema_name=schema.name if schema else "",
                            schema_encoding=schema.encoding if schema else "",
                            message_encoding=channel.message_encoding,
                            message_count=0,
                        )
                    seen[channel.id].message_count += 1
                inv.channels = list(seen.values())
                inv.total_messages = count
    except Exception as e:
        raise InspectionError(p, f"failed to read MCAP: {e}") from e

    inv.channels.sort(key=lambda c: c.topic)
    return inv


# ---------------------------------------------------------------------------
# Heuristic config generation
# ---------------------------------------------------------------------------


_IMAGE_SCHEMAS = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/Image",
    "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/CompressedImage",
    "foxglove.RawImage",
    "foxglove.CompressedImage",
    "foxglove.CompressedVideo",
}
_JOINT_STATE_SCHEMAS = {
    "sensor_msgs/msg/JointState",
    "sensor_msgs/JointState",
    "schemas.proto.JointState",
}
_POSE_SCHEMAS = {
    "geometry_msgs/msg/PoseStamped",
    "geometry_msgs/PoseStamped",
    "foxglove.PoseInFrame",
}
_TF_SCHEMAS = {
    "tf2_msgs/msg/TFMessage",
    "tf2_msgs/TFMessage",
    "foxglove.FrameTransform",
    "foxglove.FrameTransforms",
}
_FLOAT_ARRAY_SCHEMAS = {
    "std_msgs/msg/Float32MultiArray",
    "std_msgs/msg/Float64MultiArray",
    "std_msgs/Float32MultiArray",
    "std_msgs/Float64MultiArray",
}
_TASK_SCHEMAS = {
    "std_msgs/msg/String",
    "foxglove.KeyValuePair",
}

_CMD_PATTERN = re.compile(r"(cmd|command|target|commanded|desired|action|goal)", re.IGNORECASE)
_EE_PATTERN = re.compile(r"(ee|tool|tcp|wrist)", re.IGNORECASE)
_MARKER_PATTERN = re.compile(r"(/episode/|/marker|/task_description)", re.IGNORECASE)


def _topic_basename(topic: str) -> str:
    """Compact, filesystem-safe identifier from a topic path."""
    parts = [p for p in topic.strip("/").split("/") if p]
    if not parts:
        return "topic"
    # Drop common image-suffix words.
    skip = {"image", "image_raw", "image_rect", "compressed", "video_compressed", "rgb"}
    keep = [p for p in parts if p not in skip] or parts
    return "_".join(keep[:3])


@dataclass
class GenerateConfigResult:
    config: TopicConfig | None
    skipped: bool
    reason: str | None = None
    notes: list[str] = field(default_factory=list)


def generate_config(
    path: Path | str,
    inventory: MCAPInventory | None = None,
) -> GenerateConfigResult:
    """Propose a starter TopicConfig from an MCAP's channel layout.

    Returns GenerateConfigResult with `skipped=True` when no recognizable
    manipulation-style topics are present (e.g. a pure pointcloud file).
    """
    inv = inventory or inspect_mcap(path)
    notes: list[str] = []

    # Group channels by role.
    image_channels = [c for c in inv.channels if c.schema_name in _IMAGE_SCHEMAS]
    joint_channels = [c for c in inv.channels if c.schema_name in _JOINT_STATE_SCHEMAS]
    pose_channels = [c for c in inv.channels if c.schema_name in _POSE_SCHEMAS]
    float_channels = [c for c in inv.channels if c.schema_name in _FLOAT_ARRAY_SCHEMAS]
    task_channels = [c for c in inv.channels if c.schema_name in _TASK_SCHEMAS]
    marker_channels = [c for c in inv.channels if _MARKER_PATTERN.search(c.topic)]

    if not (image_channels or joint_channels or pose_channels or float_channels):
        return GenerateConfigResult(
            config=None,
            skipped=True,
            reason=(
                "no recognizable manipulation-style topics "
                "(JointState/Image/Pose/MultiArray) found"
            ),
        )

    fields: dict[str, FieldMapping] = {}

    # ---------------- state / action from joint states ----------------
    cmd_joint = [c for c in joint_channels if _CMD_PATTERN.search(c.topic)]
    obs_joint = [c for c in joint_channels if c not in cmd_joint]
    if obs_joint:
        # Highest-rate becomes the canonical observation.state.
        obs_joint.sort(key=lambda c: c.message_count, reverse=True)
        primary = obs_joint[0]
        fields["observation.state"] = FieldMapping(
            topic=primary.topic, field="position", dtype="float32"
        )
        for extra in obs_joint[1:]:
            key = f"observation.state.{_topic_basename(extra.topic)}"
            fields[key] = FieldMapping(topic=extra.topic, field="position")
            notes.append(
                f"# TODO: pick one — multiple observation joint topics found "
                f"({primary.topic} vs {extra.topic})"
            )
    if cmd_joint:
        fields["action"] = FieldMapping(
            topic=cmd_joint[0].topic, field="position", dtype="float32"
        )
        if len(cmd_joint) > 1:
            notes.append(
                "# TODO: pick one — multiple command joint topics found: "
                + ", ".join(c.topic for c in cmd_joint)
            )
    elif float_channels:
        # Fallback: a Float*MultiArray on a /cmd topic = action.
        cmd_floats = [c for c in float_channels if _CMD_PATTERN.search(c.topic)]
        if cmd_floats:
            fields["action"] = FieldMapping(topic=cmd_floats[0].topic, field="data")

    # ---------------- images ----------------
    for c in image_channels:
        key = f"observation.images.{_topic_basename(c.topic)}"
        encoding = None
        if "Video" in c.schema_name:
            encoding = "video"
        elif "Compressed" in c.schema_name:
            encoding = "jpeg"
        fields[key] = FieldMapping(
            topic=c.topic,
            **({"encoding": encoding} if encoding else {}),
        )

    # ---------------- end-effector pose ----------------
    ee_poses = [c for c in pose_channels if _EE_PATTERN.search(c.topic)]
    if ee_poses:
        fields["observation.ee_pose"] = FieldMapping(topic=ee_poses[0].topic)

    # ---------------- task description ----------------
    task_topic = next((c.topic for c in task_channels), None)

    # ---------------- episode split ----------------
    episodes = EpisodeSplit(strategy="single")
    if marker_channels:
        episodes = EpisodeSplit(
            strategy="marker", marker_topic=marker_channels[0].topic
        )

    # ---------------- sync ----------------
    sync = SyncPolicy(
        primary="observation.state" if "observation.state" in fields else None,
        method="nearest",
        max_skew_ms=50.0,
    )

    cfg = TopicConfig(
        source=Path(path),
        episodes=episodes,
        fields=fields,
        sync=sync,
    )
    if task_topic:
        cfg.task.topic = task_topic

    return GenerateConfigResult(config=cfg, skipped=False, notes=notes)
