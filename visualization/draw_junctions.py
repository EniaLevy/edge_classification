"""
Visualization utilities for rendering junctions and their branch geometry.

This module provides:
    • draw_junctions(img, junctions)
    • draw_branch_lines(img, junction)
    • draw_orientation(img, junction)

Used by:
    - main.py
    - save_outputs.py
"""

import cv2
from typing import List
from models.junction import Junction


# ---------------------------------------------------------------------
#  COLOR MAP for junction types (matching original script)
# ---------------------------------------------------------------------

JUNCTION_COLORS = {
    "l":  (255, 0, 0),       # blue
    "t":  (255, 255, 0),     # cyan/yellow-ish
    "y":  (0, 255, 255),     # teal
    "iy": (0, 255, 255),     # same as Y
    "x":  (255, 0, 255),     # magenta
    "p":  (0, 0, 255),       # red
    None: (0, 0, 255),       # default
}


# ---------------------------------------------------------------------
#  Draw branch line segments for one junction
# ---------------------------------------------------------------------

def draw_branch_lines(image, junction: Junction):
    """
    Draw the branch line segments of a junction with a color based on its type.
    """

    color = JUNCTION_COLORS.get(junction.type, (0, 0, 255))

    for ln in junction.branch_lines:
        cv2.line(
            image,
            (int(ln.x1), int(ln.y1)),
            (int(ln.x2), int(ln.y2)),
            color,
            thickness=3
        )


# ---------------------------------------------------------------------
#  Draw orientation vector for a junction (arrow)
# ---------------------------------------------------------------------

def draw_orientation(image, junction: Junction):
    """
    Draws an arrow from the junction center to its orientation endpoint.
    Only drawn if junction.orientation is not None.
    """
    if junction.orientation is None:
        return

    cv2.arrowedLine(
        image,
        (int(junction.x), int(junction.y)),
        (int(junction.orientation[0]), int(junction.orientation[1])),
        (0, 0, 0),
        thickness=2,
        tipLength=0.25
    )


# ---------------------------------------------------------------------
#  Draw full junctions (circles + branches + centers + orientation)
# ---------------------------------------------------------------------

def draw_junctions(image, junctions: List[Junction]):
    """
    Draws all junctions onto the given image.

    For each junction:
      • draw outer search radius (green)
      • draw junction core radius (green)
      • draw branch lines (junction-type dependent colors)
      • draw center point (red)
      • draw orientation arrow (if available)
    """
    for j in junctions:

        # Outer search radius
        cv2.circle(
            image,
            (int(j.x), int(j.y)),
            int(j.r),
            (0, 255, 0),
            thickness=1
        )

        # Junction radius (small inner circle)
        cv2.circle(
            image,
            (int(j.x), int(j.y)),
            int(j.junction_radius),
            (0, 255, 0),
            thickness=1
        )

        # Draw branch lines
        draw_branch_lines(image, j)

        # Draw junction center
        cv2.circle(
            image,
            (int(j.x), int(j.y)),
            1,
            (0, 0, 255),
            thickness=-1
        )

        # Orientation arrow
        draw_orientation(image, j)

    return image
