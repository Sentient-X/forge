"""MCAP format writer for Forge.

Writes Episode/Frame data to a single MCAP file. Schemas are loaded from
the bundled `.msg` definitions in ./schemas/ so the writer works without
any ROS install — `mcap` + `mcap-ros2-support` are the only deps.

Field-to-topic mapping is the inverse of TopicConfig: by default,
  observation.state               -> /forge/observation/state    (JointState)
  action                          -> /forge/action               (JointState)
  observation.images.<cam>        -> /forge/observation/images/<cam> (Image, rgb8)
  language_instruction (one-shot) -> /forge/task                 (KeyValuePair-ish via String)

Future work (PR3.1): Pose / Transform fields, h264 encoding for compressed
streams, attachment writing.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.exceptions import ConversionError, MissingDependencyError
from forge.core.models import DatasetInfo, Episode, Frame
from forge.formats.registry import FormatRegistry

SCHEMAS_DIR = Path(__file__).parent / "schemas"

# Map of canonical type name -> bundled .msg filename.
BUNDLED_SCHEMAS: dict[str, str] = {
    "sensor_msgs/JointState": "JointState.msg",
    "sensor_msgs/Image": "Image.msg",
    "sensor_msgs/CompressedImage": "CompressedImage.msg",
    "geometry_msgs/PoseStamped": "PoseStamped.msg",
    "tf2_msgs/TFMessage": "TFMessage.msg",
}


def _check_mcap() -> None:
    try:
        import mcap  # noqa: F401
        import mcap_ros2  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            dependency="mcap / mcap-ros2-support",
            feature="MCAP writing",
            install_hint='pip install "forge-robotics[mcap]"',
        ) from e


def load_bundled_schema(type_name: str) -> str:
    """Load a bundled .msg schema text by canonical type name.

    Raises FileNotFoundError if the type isn't in BUNDLED_SCHEMAS.
    """
    if type_name not in BUNDLED_SCHEMAS:
        raise FileNotFoundError(
            f"no bundled schema for {type_name!r}; "
            f"available: {sorted(BUNDLED_SCHEMAS)}"
        )
    return (SCHEMAS_DIR / BUNDLED_SCHEMAS[type_name]).read_text()


@dataclass
class MCAPWriterConfig:
    """Configuration for the MCAP writer.

    Attributes:
        chunk_compression: "zstd" | "lz4" | "none".
        image_encoding: ROS2 encoding string for raw images ("rgb8" | "bgr8" | "mono8").
        state_topic: topic for observation.state.
        action_topic: topic for action.
        image_topic_prefix: prefix for per-camera image topics (joined with cam name).
        attachments: optional list of files to embed as MCAP attachments.
    """

    chunk_compression: str = "zstd"
    image_encoding: str = "rgb8"
    state_topic: str = "/forge/observation/state"
    action_topic: str = "/forge/action"
    image_topic_prefix: str = "/forge/observation/images"
    attachments: list[Path] = field(default_factory=list)


@FormatRegistry.register_writer("mcap")
class MCAPWriter:
    """Writer for MCAP files.

    Emits one MCAP file per output_path. State/action are encoded as
    sensor_msgs/JointState (with `position` carrying the canonical vector);
    images are encoded as sensor_msgs/Image with the configured encoding.

    Example:
        >>> writer = MCAPWriter(MCAPWriterConfig(chunk_compression="zstd"))
        >>> writer.write_dataset(episodes, Path("./out.mcap"))
    """

    def __init__(self, config: MCAPWriterConfig | None = None):
        self.config = config or MCAPWriterConfig()

    @property
    def format_name(self) -> str:
        return "mcap"

    # ------------------------------------------------------------------
    # Public protocol
    # ------------------------------------------------------------------

    def write_episode(
        self,
        episode: Episode,
        output_path: Path,
        episode_index: int | None = None,
    ) -> None:
        """Write a single episode to a single MCAP file at output_path."""
        self.write_dataset(iter([episode]), output_path)

    def write_dataset(
        self,
        episodes: Iterator[Episode],
        output_path: Path,
        dataset_info: DatasetInfo | None = None,
    ) -> None:
        """Write a sequence of episodes to a single MCAP file at output_path.

        If output_path is a directory, the file is written as
        `output_path/dataset.mcap`. If it has a `.mcap` suffix, the path is
        used as-is.
        """
        _check_mcap()
        from mcap.writer import CompressionType
        from mcap_ros2.writer import Writer as Ros2Writer

        out = self._resolve_output_path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        compression = self._resolve_compression(CompressionType)

        with out.open("wb") as f:
            ros2_writer = Ros2Writer(f, compression=compression)
            schemas = self._register_schemas(ros2_writer)

            episode_count = 0
            try:
                for ep_idx, episode in enumerate(episodes):
                    self._write_episode_messages(ros2_writer, schemas, episode, ep_idx)
                    episode_count += 1
                self._write_attachments(ros2_writer)
            except Exception as e:
                raise ConversionError("source", "mcap", f"failed while writing: {e}")
            finally:
                ros2_writer.finish()

        if episode_count == 0:
            raise ConversionError("source", "mcap", "no episodes were written")

    def finalize(self, output_path: Path, dataset_info: DatasetInfo) -> None:
        """No-op — write_dataset finalizes the MCAP file in one pass."""
        return

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_output_path(self, output_path: Path) -> Path:
        out = Path(output_path)
        if out.suffix.lower() == ".mcap":
            return out
        return out / "dataset.mcap"

    def _resolve_compression(self, CompressionType: Any) -> Any:
        c = self.config.chunk_compression.lower()
        if c == "zstd":
            return CompressionType.ZSTD
        if c == "lz4":
            return CompressionType.LZ4
        if c == "none":
            return CompressionType.NONE
        raise ValueError(
            f"chunk_compression must be one of zstd/lz4/none, got {c!r}"
        )

    def _register_schemas(self, ros2_writer: Any) -> dict[str, Any]:
        """Register the bundled schemas needed for state/action/images.

        Returns a dict mapping canonical type name -> Schema record.
        """
        from mcap.records import Schema

        schemas: dict[str, Any] = {}
        for type_name in ("sensor_msgs/JointState", "sensor_msgs/Image"):
            text = load_bundled_schema(type_name)
            schema_id = ros2_writer._writer.register_schema(
                name=type_name,
                encoding="ros2msg",
                data=text.encode(),
            )
            schemas[type_name] = Schema(
                id=schema_id, name=type_name, encoding="ros2msg", data=text.encode()
            )
        return schemas

    def _write_episode_messages(
        self,
        ros2_writer: Any,
        schemas: dict[str, Any],
        episode: Episode,
        episode_index: int,
    ) -> None:
        joint_schema = schemas["sensor_msgs/JointState"]
        image_schema = schemas["sensor_msgs/Image"]

        state_topic = self._episode_topic(self.config.state_topic, episode_index)
        action_topic = self._episode_topic(self.config.action_topic, episode_index)

        for frame in episode.frames():
            log_time = self._frame_log_time(frame, episode_index)

            if frame.state is not None:
                ros2_writer.write_message(
                    topic=state_topic,
                    schema=joint_schema,
                    message=self._joint_state_message(frame, frame.state),
                    log_time=log_time,
                    publish_time=log_time,
                )

            if frame.action is not None:
                ros2_writer.write_message(
                    topic=action_topic,
                    schema=joint_schema,
                    message=self._joint_state_message(frame, frame.action),
                    log_time=log_time,
                    publish_time=log_time,
                )

            for cam_name, lazy_img in frame.images.items():
                topic = self._image_topic(cam_name, episode_index)
                ros2_writer.write_message(
                    topic=topic,
                    schema=image_schema,
                    message=self._image_message(frame, lazy_img),
                    log_time=log_time,
                    publish_time=log_time,
                )

    def _frame_log_time(self, frame: Frame, episode_index: int) -> int:
        """Frame timestamps may be small floats (e.g. seconds-from-start) or
        absolute Unix nanos. Either way we just need monotonic per-episode
        ordering. Use a synthetic time anchored per episode if no timestamp."""
        if frame.timestamp is not None:
            return int(frame.timestamp * 1e9)
        # Fallback: 30Hz from an episode-anchored base.
        return int(episode_index * 1e10 + frame.index * (1e9 / 30.0))

    def _joint_state_message(self, frame: Frame, vec: np.ndarray) -> dict[str, Any]:
        arr = np.asarray(vec, dtype=np.float64).flatten()
        sec = int(frame.timestamp) if frame.timestamp is not None else 0
        nsec = int(((frame.timestamp or 0.0) - sec) * 1e9) if frame.timestamp is not None else 0
        return {
            "header": {
                "stamp": {"sec": sec, "nanosec": nsec},
                "frame_id": "",
            },
            "name": [],  # joint names not preserved in the canonical IR
            "position": arr.tolist(),
            "velocity": [],
            "effort": [],
        }

    def _image_message(self, frame: Frame, lazy_img: Any) -> dict[str, Any]:
        img = np.asarray(lazy_img.load())
        if img.ndim == 2:
            h, w = img.shape
            channels = 1
        elif img.ndim == 3:
            h, w, channels = img.shape
        else:
            raise ValueError(f"image must be 2D or 3D, got shape {img.shape}")

        encoding = self.config.image_encoding
        if channels == 1 and "mono" not in encoding:
            encoding = "mono8"
        elif channels == 3 and "mono" in encoding:
            encoding = "rgb8"

        sec = int(frame.timestamp) if frame.timestamp is not None else 0
        nsec = int(((frame.timestamp or 0.0) - sec) * 1e9) if frame.timestamp is not None else 0
        return {
            "header": {
                "stamp": {"sec": sec, "nanosec": nsec},
                "frame_id": "",
            },
            "height": int(h),
            "width": int(w),
            "encoding": encoding,
            "is_bigendian": 0,
            "step": int(w * channels),
            "data": img.astype(np.uint8, copy=False).tobytes(),
        }

    def _episode_topic(self, base: str, episode_index: int) -> str:
        # We embed the episode index into the topic so re-readers can split
        # without a marker-topic config.
        return f"{base}/ep_{episode_index:04d}"

    def _image_topic(self, cam_name: str, episode_index: int) -> str:
        return f"{self.config.image_topic_prefix}/{cam_name}/ep_{episode_index:04d}"

    def _write_attachments(self, ros2_writer: Any) -> None:
        if not self.config.attachments:
            return
        from mcap.records import Attachment

        for att_path in self.config.attachments:
            p = Path(att_path)
            if not p.exists():
                raise ConversionError(
                    "source", "mcap", f"attachment not found: {p}"
                )
            data = p.read_bytes()
            ros2_writer._writer.add_attachment(
                create_time=0,
                log_time=0,
                name=p.name,
                media_type=_guess_media_type(p),
                data=data,
            )


def _guess_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".urdf": "application/xml",
        ".xml": "application/xml",
        ".json": "application/json",
        ".yaml": "application/x-yaml",
        ".yml": "application/x-yaml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(suffix, "application/octet-stream")
