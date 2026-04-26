"""First-class MCAP format support for Forge.

MCAP is a serialization-agnostic container; Forge supports ROS2 (CDR) and
Foxglove/Protobuf message encodings via the optional `[mcap]` extra.

Public API:
    MCAPReader, MCAPWriter, MCAPWriterConfig
    TopicConfig, load_config, validate_config
    inspect_mcap, generate_config
"""

from __future__ import annotations

from forge.formats.mcap.attachments import (
    AttachmentRecord,
    extract_attachments,
    list_attachments,
)
from forge.formats.mcap.inspect import generate_config, inspect_mcap
from forge.formats.mcap.reader import MCAPReader
from forge.formats.mcap.topic_config import TopicConfig, load_config, validate_config
from forge.formats.mcap.writer import MCAPWriter, MCAPWriterConfig

__all__ = [
    "AttachmentRecord",
    "MCAPReader",
    "MCAPWriter",
    "MCAPWriterConfig",
    "TopicConfig",
    "extract_attachments",
    "generate_config",
    "inspect_mcap",
    "list_attachments",
    "load_config",
    "validate_config",
]
