"""Pre-decode an h264/h265 packet stream into a list of RGB frames.

MCAP camera topics that carry `sensor_msgs/CompressedImage` with
`format='h264'` (or `foxglove.CompressedVideo`) store one NAL packet per
message. Each packet is meaningless on its own — the codec needs SPS/PPS
parameter sets and inter-frame references. So we feed the whole sequence
through a stateful PyAV CodecContext once and cache the decoded RGB arrays.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

H264_FORMATS = {"h264", "h265", "hevc", "video", "h264_nal"}


def looks_like_video_format(fmt: str | None) -> bool:
    if not fmt:
        return False
    f = fmt.lower()
    return any(token in f for token in H264_FORMATS)


def decode_packet_stream(
    packets: list[bytes], codec: str = "h264"
) -> list[NDArray[Any]]:
    """Decode an ordered list of h264/h265 NAL packets into RGB uint8 frames.

    Returns one decoded frame per *output* of the codec (which may be fewer
    than the number of packets, since some packets are SPS/PPS only). The
    caller is responsible for aligning frame indices.

    Falls back to returning an empty list if PyAV isn't installed or the
    stream can't be opened.
    """
    if not packets:
        return []
    try:
        import av
    except ImportError:
        logger.warning("mcap: video decode requested but PyAV is not installed")
        return []

    try:
        ctx = av.codec.CodecContext.create(codec, "r")
    except Exception as e:
        logger.warning("mcap: failed to create %s codec context: %s", codec, e)
        return []

    frames: list[NDArray[Any]] = []
    for raw in packets:
        try:
            packet = av.packet.Packet(raw)
            for av_frame in ctx.decode(packet):
                frames.append(av_frame.to_ndarray(format="rgb24"))
        except Exception as e:
            logger.debug("mcap: skipping packet: %s", e)
            continue

    # Flush any buffered frames.
    try:
        for av_frame in ctx.decode(None):
            frames.append(av_frame.to_ndarray(format="rgb24"))
    except Exception:
        pass

    return frames
