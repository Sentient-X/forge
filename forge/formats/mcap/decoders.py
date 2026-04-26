"""Unified message decoder for MCAP — ROS2 CDR + Foxglove Protobuf.

The MCAP container holds raw bytes; each channel has a schema describing the
message encoding. This module wires both `mcap-ros2-support` and
`mcap-protobuf-support` decoder factories into a single reader so callers
get a uniform `DecodedMessage` stream regardless of payload encoding.

Field extractors operate on attribute access — works for both ROS2 dataclasses
and protobuf objects.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.core.exceptions import MissingDependencyError


def _check_mcap() -> None:
    try:
        import mcap  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            dependency="mcap",
            feature="MCAP format support",
            install_hint='pip install "forge-robotics[mcap]"',
        ) from e


@dataclass
class DecodedMessage:
    """One decoded MCAP message ready for field extraction."""

    topic: str
    timestamp_ns: int
    publish_time_ns: int
    schema_name: str
    schema_encoding: str
    message: Any


def _build_decoder_factories() -> list[Any]:
    """Build all available decoder factories. Missing ones are skipped silently
    so a Protobuf-only file still works without mcap-ros2-support installed."""
    factories: list[Any] = []
    try:
        from mcap_ros2.decoder import DecoderFactory as Ros2Factory

        factories.append(Ros2Factory())
    except ImportError:
        pass
    try:
        from mcap_protobuf.decoder import DecoderFactory as ProtoFactory

        factories.append(ProtoFactory())
    except ImportError:
        pass
    return factories


def iter_decoded(
    path: Path,
    topics: set[str] | None = None,
) -> Iterator[DecodedMessage]:
    """Stream decoded messages from an MCAP file in log-time order.

    Args:
        path: MCAP file.
        topics: if not None, only emit messages whose topic is in this set.

    Yields:
        DecodedMessage in log_time order. Messages whose schema encoding has
        no available decoder (e.g. jsonschema with no factory installed) are
        silently skipped — call `inspect_mcap` first to see encodings.
    """
    _check_mcap()
    from mcap.reader import make_reader

    factories = _build_decoder_factories()
    if not factories:
        raise MissingDependencyError(
            dependency="mcap-ros2-support / mcap-protobuf-support",
            feature="MCAP message decoding",
            install_hint='pip install "forge-robotics[mcap]"',
        )

    topic_filter = list(topics) if topics else None

    with Path(path).open("rb") as f:
        reader = make_reader(f, decoder_factories=factories)
        for schema, channel, message, decoded in reader.iter_decoded_messages(
            topics=topic_filter
        ):
            if schema is None or decoded is None:
                continue
            yield DecodedMessage(
                topic=channel.topic,
                timestamp_ns=message.log_time,
                publish_time_ns=message.publish_time,
                schema_name=schema.name,
                schema_encoding=schema.encoding,
                message=decoded,
            )
