"""Episode segmentation via PELT changepoint detection.

Detects phase transitions in proprioception signals to split episodes
into meaningful segments (sub-skills, regime changes).

Usage::

    from forge.segment import SegmentAnalyzer

    analyzer = SegmentAnalyzer(penalty="bic", cost_model="rbf")
    report = analyzer.segment_dataset("./bridge_v2")
    report.to_json("segments.json")
"""

from forge.segment.analyzer import SegmentAnalyzer
from forge.segment.config import SegmentConfig
from forge.segment.labeler import PhaseLabeler
from forge.segment.models import EpisodeSegmentation, SegmentationReport

__all__ = [
    "SegmentAnalyzer",
    "SegmentConfig",
    "EpisodeSegmentation",
    "PhaseLabeler",
    "SegmentationReport",
]
