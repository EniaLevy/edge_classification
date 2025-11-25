"""
Visualization utilities for rendering line segments.

This module provides:
    • draw_lines(img, lines, color, thickness)
    • draw_colored_edges(img, straight_edges)

It is used by:
    - main.py
    - visualization.save_outputs
    - detectors that want temporary visualization
"""

import cv2
from typing import List, Tuple
from models.line import Line
from models.straight_edge import StraightEdge


# ---------------------------------------------------------------------
#  BASIC: Draw a list of line segments in a single color
# ---------------------------------------------------------------------

def draw_lines(
    image,
    lines: List[Line],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
):
    """
    Draws a simple list of Line objects onto an image.

    Args:
        image: BGR numpy array (modified in-place)
        lines: list of Line objects
        color: (B, G, R)
        thickness: pixel width
    """
    for ln in lines:
        cv2.line(
            image,
            (int(ln.x1), int(ln.y1)),
            (int(ln.x2), int(ln.y2)),
            color,
            thickness
        )
    return image


# ---------------------------------------------------------------------
#  HIGH-LEVEL: Draw StraightEdge groups with classification colors
# ---------------------------------------------------------------------

def draw_colored_edges(
    image,
    edges: List[StraightEdge],
    thickness: int = 2
):
    """
    Draws full StraightEdge objects with colors based on their type:

        ob → blue    (255, 0, 0)
        sc → green   (0, 255, 0)
        rc → red     (0, 0, 255)
        None → white (255, 255, 255)

    Args:
        image: BGR numpy array
        edges: list of StraightEdge objects (already classified)
        thickness: pixel width
    """

    for e in edges:
        # Assign color based on type
        match e.type:
            case "ob":  color = (255, 0, 0)       # blue
            case "sc":  color = (0, 255, 0)       # green
            case "rc":  color = (0, 0, 255)       # red
            case _:     color = (255, 255, 255)   # white

        for seg in e.line_segments:
            cv2.line(
                image,
                (int(seg.x1), int(seg.y1)),
                (int(seg.x2), int(seg.y2)),
                color=color,
                thickness=thickness
            )

    return image
