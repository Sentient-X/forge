# `forge.formats.mcap`

First-class MCAP support for Forge — reader **and** writer, ROS2 CDR + Foxglove
Protobuf decoding, no ROS install required.

## Why MCAP gets its own package

MCAP is a **serialization-agnostic container**, not a ROS-only format. It can
hold ROS2 CDR, Protobuf (Foxglove), JSON, or raw bytes — often in the same
file. The legacy rosbag reader could open `.mcap` files but only via the
`rosbags` library, which assumes ROS2 message types. That breaks on Foxglove
Protobuf payloads (e.g. the Trossen Transfer Cube fixture).

This package treats MCAP as its own first-class spoke in the hub-and-spoke
architecture: registered separately, prioritized above `rosbag` for `.mcap`
files, and built on the official `mcap` + `mcap-ros2-support` +
`mcap-protobuf-support` libraries.

## Install

```bash
pip install -e ".[mcap]"
```

This pulls in `mcap`, `mcap-ros2-support`, and `mcap-protobuf-support` only —
no `rclpy`, no `rosbag2_py`, no ROS install.

## Quick start

```bash
# Inspect — channel/schema list, message counts, profile, attachments.
forge inspect sample_data/mcap/trossen_transfer_cube.mcap

# Convert MCAP -> LeRobot v3 (auto-detect topic mapping).
forge convert teleop.mcap ./out --format lerobot-v3

# Convert any source format -> MCAP.
forge convert ./lerobot_dataset out.mcap --format mcap

# Visualize (requires the rerun viewer binary).
forge visualize teleop.mcap --backend rerun
```

## Topic config (YAML)

MCAP topics are not self-describing about which channel is "state" vs "action",
so Forge uses a YAML topic config to drive both directions:

```yaml
source: ./teleop_session.mcap

episodes:
  strategy: marker             # marker | time_gap | segment | single
  marker_topic: /episode/start
  min_length_frames: 30
  drop_first_n_frames: 5

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

sync:
  primary: observation.state
  method: nearest              # nearest | interpolate | hold
  max_skew_ms: 50

attachments:
  - name: robot_urdf
    path: ./allegro.urdf
    media_type: application/xml

task:
  description: "stack red on blue"
  # OR — read it from the MCAP itself: topic: /task_description
```

`generate_config()` produces a starter YAML using auto-detection heuristics
(see `inspect.py`):

```python
from forge.formats.mcap import generate_config
from forge.formats.mcap.topic_config import dump_config

result = generate_config("sample_data/mcap/trossen_transfer_cube.mcap")
if not result.skipped:
    dump_config(result.config, "trossen.yaml")
```

## Episode strategies

| Strategy | Splits when |
|---|---|
| `single` | One episode covering all primary timestamps |
| `marker` | Each marker-topic message starts a new episode |
| `time_gap` | Consecutive primary timestamps differ by > `time_gap_seconds` |
| `segment` | PELT changepoint detection on a numeric signal stream |

Post-filters (`min_length_frames`, `drop_first_n_frames`) apply to all
strategies.

## Sync methods

Primary topic timestamps drive frame boundaries — one Episode frame per
primary message. Secondary fields are aligned via:

- **`nearest`** — closest timestamp regardless of side
- **`interpolate`** — linear interp for numeric arrays; falls back to nearest for images / non-numeric
- **`hold`** — most recent value at-or-before the primary timestamp (zero-order hold)

If skew exceeds `max_skew_ms`, an aggregated warning is emitted at end of
episode (one per field, with count + max skew). Frames are dropped only when
skew exceeds **10× the threshold**.

## Module layout

```
forge/formats/mcap/
├── __init__.py           # Public API
├── reader.py             # MCAPReader (FormatReader)
├── writer.py             # MCAPWriter (FormatWriter), MCAPWriterConfig
├── inspect.py            # inspect_mcap, generate_config heuristics
├── topic_config.py       # YAML schema (dataclass), load/dump/validate
├── sync.py               # nearest / hold / interpolate / align / align_stream
├── episode_split.py      # single / marker / time_gap / segment
├── decoders.py           # Unified ROS2 CDR + Protobuf decoder
├── extractors.py         # JointState position/velocity, Image bytes, strings
├── video_decode.py       # h264/h265 packet stream -> RGB frames (PyAV)
├── attachments.py        # list_attachments, extract_attachments
├── schemas/              # Bundled .msg defs for self-describing writes
│   ├── JointState.msg
│   ├── Image.msg
│   ├── CompressedImage.msg
│   ├── PoseStamped.msg
│   └── TFMessage.msg
└── README.md             # (this file)
```

## Test corpus

Three fixtures live under [`sample_data/mcap/`](../../../sample_data/mcap/),
sourced from the [Rerun project](https://github.com/rerun-io/rerun/tree/main/tests/assets/mcap):

| File | Profile | Encoding | Notes |
|---|---|---|---|
| `r2b_galileo.mcap` | `ros2` | CDR | Stereo cameras + IMU + battery (NVIDIA R2B) |
| `supported_ros2_messages.mcap` | `ros2` | CDR | Wide coverage of `sensor_msgs/*` types |
| `trossen_transfer_cube.mcap` | (none) | Protobuf | Bimanual teleop with foxglove video |

The full inventory is regenerated by `python scripts/mcap_inventory.py` →
[`tests/fixtures/mcap/INVENTORY.md`](../../../tests/fixtures/mcap/INVENTORY.md).

## Current limitations

- **h264 / h265 image streams need PyAV** (already in `[video]` extra). When
  PyAV is missing the reader falls back to raw NAL bytes and the visualizer
  will reject them — install with `pip install -e ".[mcap,video]"`.
- **Writer encodes images as `sensor_msgs/Image` (raw rgb8)** — round-tripping
  an h264 input produces a much larger output. h264 re-encoding on write
  hasn't shipped yet.
- **Joint names aren't preserved** — the canonical Frame model has only a
  state vector. Names land in `metadata` if you set them explicitly.

## Design notes

- **Frame.extras** — fields outside the canonical state/action/images slots
  (e.g. `observation.ee_pose`, secondary `observation.state.velocity`) land in
  `Frame.extras: dict[str, Any]`. Existing readers don't populate it; existing
  consumers ignore it.
- **No silent guessing in topic resolution** — when in doubt, raise. The
  exception message points to the offending YAML field.
- **Don't break the rosbag path** — rosbag reader still claims `.mcap` if the
  `[mcap]` extra isn't installed. The detection priority puts `mcap` above
  `rosbag` so installed users always get the new path.
