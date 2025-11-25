"""
Edge classifier for StraightEdge objects.

This module provides:
    • group_into_straight_edges(lines)
    • classify_edge_objects(straight_edges, junctions)
    • build_predicted_map(straight_edges, image_shape)
    • build_classified_image(straight_edges, base_image)
"""

import numpy as np
import pybresenham as bres
import cv2

from models.straight_edge import StraightEdge
from models.line import Line
from models.junction import Junction
from utils.geometry import calculateAngle, checkIfParallel
from config import get_active_params, COLOR_OB, COLOR_SC, COLOR_RC, COLOR_OTHER


# ========================================================================
# 1. GROUP LINES INTO STRAIGHT EDGES (parallel + close proximity)
# ========================================================================

def group_into_straight_edges(lines):
    """
    Groups nearly-parallel lines into StraightEdge objects,
    replicating the BFS grouping:

        Two lines belong in the same group if BOTH:
            - abs(angle) < ANGLE_THRESHOLD/2  (or near 180)
            - distance between them < DISTANCE_THRESHOLD/2

    Returns:
        List[StraightEdge]
    """
    params = get_active_params()
    distance_threshold = params["DISTANCE_THRESHOLD"]
    angle_threshold = params["ANGLE_THRESHOLD"]

    visited = set()
    edges = []

    for i, line in enumerate(lines):
        if line in visited:
            continue

        group = []
        queue = [line]

        while queue:
            L = queue.pop(0)
            if L in group:
                continue
            group.append(L)
            visited.add(L)

            for other in lines:
                if other in visited:
                    continue

                # angle must be extremely close
                ang = calculateAngle(L, other)
                if ang < angle_threshold / 2 or abs(ang - 180) < angle_threshold / 2:

                    # distance check using endpoints as original did
                    d1 = 0
                    d2 = 0
                    if not L.pointIsOnLine(other.a1):
                        d1 = L.distanceFromPointToLineSegment(other.a1)
                    if not L.pointIsOnLine(other.a2):
                        d2 = L.distanceFromPointToLineSegment(other.a2)

                    if d1 <= distance_threshold / 2 or d2 <= distance_threshold / 2:
                        queue.append(other)

        # Create straight edge object
        edge = StraightEdge()
        for seg in group:
            edge.addLineSeg(seg)
        edges.append(edge)

    return edges


# ========================================================================
# 2. CLASSIFY EDGES USING JUNCTIONS
# ========================================================================

def classify_edge_objects(straight_edges, junctions):
    """
    Logic:
        - attach junctions to straight edges
        - assign branch types (ts, tb, ps, pb, ys, yb)
        - apply scoring (ob, sc, rc)
        - set final edge.type
    """
    params = get_active_params()

    for edge in straight_edges:
        # --------------------------
        # Find junctions that use any line in this edge
        # --------------------------
        for j in junctions:
            for line in j.line_segments:
                if line in edge.line_segments:
                    edge.addJunction(j)

                    # Branch type assignment
                    _assign_branch_type(j, line, edge)
                    break

        # --------------------------
        # Score & classify
        # --------------------------
        edge.calculateScore()
        edge.checkType()

    return straight_edges


# ========================================================================
# 2a. BRANCH TYPE CLASSIFICATION (ts, tb, ps, pb, ys, yb)
# ========================================================================

def _assign_branch_type(j: Junction, line: Line, edge: StraightEdge):
    """
    Branch-type logic:

        - If junction has spine_branches:
            t-branch: ts / tb
            p-branch: ps / pb
            y-branch: ys / yb

        - Otherwise skip (non-spine / unclassified junction)
    """
    if not j.spine_branches:
        return

    spine_check = False
    for branch_idx in j.spine_branches:
        if checkIfParallel(j.branch_lines[branch_idx], line):
            spine_check = True
            break

    if spine_check:
        match j.type:
            case "t":  edge.branch_types.append("ts")
            case "p":  edge.branch_types.append("ps")
            case "iy": edge.branch_types.append("ys")
    else:
        match j.type:
            case "t":  edge.branch_types.append("tb")
            case "p":  edge.branch_types.append("pb")
            case "iy": edge.branch_types.append("yb")


# ========================================================================
# 3. PREDICTION MAP (rc, sc, ob)
# ========================================================================

def build_predicted_map(straight_edges, shape_hw):
    """
    Builds the pixel-wise prediction map:
        1 = ob (blue)
        2 = sc (green)
        3 = rc (red)

    Approach:
        - iterate through each segment
        - set predicted[y,x] = class_label
    """
    h, w = shape_hw
    predicted = np.zeros((h, w), dtype=np.uint8)

    for edge in straight_edges:
        class_label = _edge_type_to_label(edge.type)

        for seg in edge.line_segments:
            points = bres.line(int(seg.x1), int(seg.y1),
                               int(seg.x2), int(seg.y2))
            for x, y in points:
                if 0 <= x < w and 0 <= y < h:
                    predicted[y, x] = class_label

    return predicted


def _edge_type_to_label(t):
    if t == "ob":
        return 1
    if t == "sc":
        return 2
    if t == "rc":
        return 3
    return 0


# ========================================================================
# 4. DRAWING COLORED STRAIGHT EDGES (matching original output style)
# ========================================================================

def build_classified_image(straight_edges, base_image):
    """
    Returns a copy of the image where straight edges are drawn in color:
        ob → blue
        sc → green
        rc → red
        other → white
    """
    out = base_image.copy()

    for e in straight_edges:
        color = _edge_type_to_color(e.type)
        for seg in e.line_segments:
            cv2.line(
                out,
                (int(seg.x1), int(seg.y1)),
                (int(seg.x2), int(seg.y2)),
                color,
                thickness=2
            )
    return out


def _edge_type_to_color(t):
    if t == "ob":
        return COLOR_OB
    if t == "sc":
        return COLOR_SC
    if t == "rc":
        return COLOR_RC
    return COLOR_OTHER
