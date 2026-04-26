# Forge Architecture

## Overview

Forge is designed around a simple principle: **any format in, any format out**, using a canonical intermediate representation.

## Core Data Flow

```mermaid
flowchart LR
    subgraph Input Formats
        RLDS[RLDS]
        LR2[LeRobot v2]
        LR3[LeRobot v3]
        Zarr[Zarr]
        HDF5[HDF5]
        ROS[ROS bags]
    end

    subgraph Forge Core
        Reader[Format Reader]
        Episode[Episode/Frame Model]
        Writer[Format Writer]
    end

    subgraph Output
        Out[LeRobot v3]
        Viz[Visualizer]
        Inspect[Inspector]
        Quality[Quality Analyzer]
    end

    RLDS --> Reader
    LR2 --> Reader
    LR3 --> Reader
    Zarr --> Reader
    HDF5 --> Reader
    ROS --> Reader

    Reader --> Episode
    Episode --> Writer
    Episode --> Viz
    Episode --> Inspect
    Episode --> Quality

    Writer --> Out
```

## Canonical Data Model

All formats are normalized to this intermediate representation:

```mermaid
classDiagram
    class Episode {
        +str episode_id
        +dict metadata
        +str language_instruction
        +frames() Iterator~Frame~
    }

    class Frame {
        +int index
        +float timestamp
        +dict~str,LazyImage~ images
        +ndarray state
        +ndarray action
        +bool is_first
        +bool is_last
    }

    class LazyImage {
        +int height
        +int width
        +int channels
        +load() ndarray
    }

    Episode "1" --> "*" Frame : contains
    Frame "1" --> "*" LazyImage : has cameras
```

## Format Registry

Formats register themselves using decorators:

```mermaid
flowchart TB
    subgraph Registration
        Dec1["@register_reader('hdf5')"]
        Dec2["@register_writer('lerobot-v3')"]
    end

    subgraph FormatRegistry
        Readers[readers dict]
        Writers[writers dict]
        Detect[detect_format]
    end

    subgraph Usage
        GetR[get_reader]
        GetW[get_writer]
    end

    Dec1 --> Readers
    Dec2 --> Writers
    Readers --> Detect
    Detect --> GetR
    Writers --> GetW
```

## CLI Command Flow

```mermaid
flowchart TD
    subgraph Commands
        Inspect[forge inspect]
        Convert[forge convert]
        Visualize[forge visualize]
        QualityCmd[forge quality]
        FilterCmd[forge filter]
        Hub[forge hub]
    end

    subgraph Hub Integration
        HF[HuggingFace Hub]
        Cache[~/.cache/forge/datasets]
    end

    subgraph Core
        Registry[FormatRegistry]
        Reader[Reader]
        Writer[Writer]
        Viewer[UnifiedViewer]
    end

    Inspect --> Registry
    Convert --> Registry
    Visualize --> Registry
    QualityCmd --> Registry
    FilterCmd --> Registry

    Hub --> HF
    HF --> Cache
    Cache --> Registry

    Registry --> Reader
    Registry --> Writer
    Reader --> Viewer
```

## Conversion Pipeline

```mermaid
sequenceDiagram
    participant CLI
    participant Registry
    participant Reader
    participant Writer
    participant Disk

    CLI->>Registry: detect_format(source)
    Registry-->>CLI: "hdf5"
    CLI->>Registry: get_reader("hdf5")
    Registry-->>CLI: HDF5Reader
    CLI->>Registry: get_writer("lerobot-v3")
    Registry-->>CLI: LeRobotV3Writer

    CLI->>Reader: read_episodes(source)
    loop For each episode
        Reader-->>CLI: Episode
        CLI->>Writer: write_episode(episode)
        Writer->>Disk: parquet + mp4
    end
    CLI->>Writer: finalize()
    Writer->>Disk: meta/info.json
```

## Project Structure

```
forge/
├── cli.py                 # CLI commands (inspect, convert, visualize, quality, filter, hub)
├── core/
│   ├── models.py          # Episode, Frame, LazyImage
│   ├── protocols.py       # FormatReader, FormatWriter interfaces
│   └── exceptions.py      # Custom exceptions
├── formats/
│   ├── registry.py        # FormatRegistry
│   ├── rlds/              # RLDS reader
│   ├── lerobot_v2/        # LeRobot v2 reader
│   ├── lerobot_v3/        # LeRobot v3 reader + writer
│   ├── zarr/              # Zarr reader
│   ├── hdf5/              # HDF5 reader (robomimic, ACT/ALOHA)
│   ├── mcap/              # MCAP reader + writer (ROS2 CDR + Foxglove Protobuf)
│   └── rosbag/            # ROS1 bag + ROS2 SQLite3 reader
├── hub/
│   ├── url.py             # hf:// URL parsing
│   └── download.py        # Dataset downloading
├── convert/
│   └── converter.py       # Conversion orchestration
├── inspect/
│   └── inspector.py       # Dataset inspection
├── quality/
│   ├── config.py          # QualityConfig (thresholds, weights)
│   ├── metrics.py         # Individual metric functions
│   ├── analyzer.py        # QualityAnalyzer orchestrator
│   └── models.py          # EpisodeQuality, QualityReport
├── filter/
│   └── engine.py          # FilterEngine, FilterConfig, FilterResult
└── visualize/
    └── unified_viewer.py  # Format-agnostic viewer
```

## Adding a New Format

1. Create format directory: `forge/formats/myformat/`

2. Implement reader:
```python
@FormatRegistry.register_reader("myformat")
class MyFormatReader:
    @classmethod
    def can_read(cls, path: Path) -> bool:
        # Detection logic
        ...

    def inspect(self, path: Path) -> DatasetInfo:
        # Return dataset metadata
        ...

    def read_episodes(self, path: Path) -> Iterator[Episode]:
        # Yield episodes with lazy frame loading
        ...
```

3. Register in `forge/formats/__init__.py`:
```python
try:
    from forge.formats import myformat
except ImportError:
    pass
```

4. Add to detection priority in `registry.py` if needed

That's it! Your format now works with:
- `forge inspect`
- `forge convert ... --format lerobot-v3`
- `forge visualize`
