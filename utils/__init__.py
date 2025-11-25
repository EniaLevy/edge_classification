"""
Utility Functions

Provides geometry operations, Bresenham wrappers, clustering,
image I/O utilities, and helper functions used across detectors.
"""

from .geometry import (
    line_intersect,
    calculateAngle,
    checkIfParallel,
    getPixelNeighborhood,
    checkBranchParallels,
)
from .bresenham_utils import bres_line, bres_circle
from .clustering import (
    cluster_points_radius,
    cluster_objects_radius,
    group_by_connectivity,
    deduplicate_close_points,
    pop_items,
)
from .image_io import load_images, extract_numeric_id, ensure_output_dir, save_image

__all__ = [
    "line_intersect",
    "calculateAngle",
    "checkIfParallel",
    "getPixelNeighborhood",
    "checkBranchParallels",
    "bres_line",
    "bres_circle",
    "cluster_points_radius",
    "cluster_objects_radius",
    "group_by_connectivity",
    "deduplicate_close_points",
    "pop_items",
    "load_images",
    "extract_numeric_id",
    "ensure_output_dir",
    "save_image",
]
