"""
Utility wrappers around the pybresenham library.

This module provides:
    • bres_line(x1, y1, x2, y2)
    • bres_circle(cx, cy, r)

These functions return lists of (x, y) integer pixel coordinates.

Reason for existence:
    - isolates the external dependency (pybresenham)
    - lets other modules import Bresenham functionality uniformly
    - provides a safe fallback behavior if pybresenham is missing
"""

from typing import List, Tuple

try:
    import pybresenham as bres
except ImportError:
    bres = None


# -----------------------------------------------------------
#   Line drawing wrapper
# -----------------------------------------------------------

def bres_line(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
    """
    Returns a list of integer pixel coordinates forming a Bresenham line.

    If pybresenham is not installed, falls back to a simple linear interpolation.
    """
    if bres is not None:
        return [(int(x), int(y)) for x, y in bres.line(x1, y1, x2, y2)]

    # Fallback: simple interpolation (non-optimal but adequate)
    points = []
    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))

    if steps == 0:
        return [(x1, y1)]

    x_inc = dx / steps
    y_inc = dy / steps

    x, y = x1, y1
    for _ in range(steps + 1):
        points.append((int(round(x)), int(round(y))))
        x += x_inc
        y += y_inc

    return points


# -----------------------------------------------------------
#   Circle wrapper
# -----------------------------------------------------------

def bres_circle(cx: int, cy: int, r: int) -> List[Tuple[int, int]]:
    """
    Returns a list of integer pixel coordinates forming a Bresenham circle.

    If pybresenham is not installed, returns a simple rasterized circle.
    """
    if r < 0:
        return []

    if bres is not None:
        return [(int(x), int(y)) for x, y in bres.circle(cx, cy, r)]

    # Fallback approximation
    points = []
    for angle in range(360):
        rad = angle * 3.1415926535 / 180.0
        x = cx + int(r * (float)(__import__("math").cos(rad)))
        y = cy + int(r * (float)(__import__("math").sin(rad)))
        points.append((x, y))
    return points

# -----------------------------------------------------------
#   Line wrapper
# -----------------------------------------------------------

def draw_line(x1: int, y1: int, x2: int, y2: int):
    """
    Backwards-compatible alias for bres_line, used by line_detector.
    """
    return bres_line(x1, y1, x2, y2)
