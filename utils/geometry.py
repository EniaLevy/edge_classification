"""
This module provides:
    - line_intersect
    - calculateAngle
    - checkIfParallel  (with auto threshold from config)
    - getPixelNeighborhood
    - checkBranchParallels
"""

import math

from config import get_active_params


# ----------------------------------------------------------------------
#  LINE INTERSECTION (UNMODIFIED ORIGINAL LOGIC)
# ----------------------------------------------------------------------

def line_intersect(m1, b1, m2, b2):
    """
    Compute intersection point between two lines defined by:
        y = m1 x + b1
        y = m2 x + b2

    Returns:
        (x, y) or None if parallel (m1 == m2)
    """
    if m1 == m2:
        return None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y


# ----------------------------------------------------------------------
#  ANGLE BETWEEN TWO LINES (SLOPE-BASED)
# ----------------------------------------------------------------------

def calculateAngle(line1, line2):
    """
    Returns the absolute angle (degrees) between two Line objects,
    computed from their slopes.
    """
    s1 = line1.m
    s2 = line2.m

    if s1 == s2:
        return 0

    try:
        # tan(theta) = |(m2 - m1) / (1 + m1*m2)|
        return abs(math.degrees(math.atan((s2 - s1) / (1 + (s2 * s1)))))
    except ZeroDivisionError:
        # Handles infinite/vertical slope cases
        angle = abs(1 + (s1 * s2))
        denom = math.sqrt(1 + s1 ** 2) * math.sqrt(1 + s2 ** 2)
        if denom == 0:
            return 0
        angle /= denom
        # acos output is in radians
        return math.degrees(math.acos(angle))


# ----------------------------------------------------------------------
#  PARALLEL CHECK (AUTO-THRESHOLD FROM CONFIG)
# ----------------------------------------------------------------------

def checkIfParallel(l1, l2, angle_threshold=None):
    """
    Returns True if two Line objects are parallel or anti-parallel.
    """
    if angle_threshold is None:
        params = get_active_params()
        angle_threshold = params["ANGLE_THRESHOLD"]

    ang = calculateAngle(l1, l2)
    return (ang < angle_threshold) or (abs(ang - 180) < angle_threshold)


# ----------------------------------------------------------------------
#  PIXEL NEIGHBORHOOD (IDENTICAL TO ORIGINAL)
# ----------------------------------------------------------------------

def getPixelNeighborhood(center, size=1):
    """
    Returns an (2*size+1)-square of pixel coordinates centered on 'center'.

    Example: size = 1 → 3×3 neighborhood
    """
    cx, cy = center
    pts = []
    for dx in range(-size, size + 1):
        for dy in range(-size, size + 1):
            pts.append((cx + dx, cy + dy))
    return pts


# ----------------------------------------------------------------------
#  BRANCH PARALLELITY CHECK
# ----------------------------------------------------------------------

def checkBranchParallels(branch_lines, angle_threshold=None):
    """
    Checks how many branch-line pairs are parallel.

    Input:
        branch_lines: list of Line objects (branch proxies)
    Output:
        count: number of parallel pairs
        idx_pairs: list of (i, j) index tuples
    """
    if angle_threshold is None:
        params = get_active_params()
        angle_threshold = params["ANGLE_THRESHOLD"]

    count = 0
    idx_pairs = []

    for i in range(len(branch_lines)):
        for j in range(i + 1, len(branch_lines)):
            if checkIfParallel(branch_lines[i], branch_lines[j], angle_threshold):
                count += 1
                idx_pairs.append((i, j))

    return count, idx_pairs
