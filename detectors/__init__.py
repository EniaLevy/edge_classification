"""
Detectors Package

Contains the main detection modules used in the edge classification pipeline:
- Line detection
- Junction candidate generation & merging
- Branch extraction
- Straight-edge classification
"""

from .line_detector import detect_lines, build_line_id_map
from .junction_detector import (
    detect_junction_candidates,
    merge_junctions,
    classify_junctions,
)
from .branch_finder import extract_branches
from .edge_classifier import (
    group_into_straight_edges,
    classify_edge_objects,
    build_predicted_map,
)

__all__ = [
    "detect_lines",
    "build_line_id_map",
    "detect_junction_candidates",
    "merge_junctions",
    "classify_junctions",
    "extract_branches",
    "group_into_straight_edges",
    "classify_edge_objects",
    "build_predicted_map",
]
