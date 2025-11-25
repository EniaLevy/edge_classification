"""
This module provides:
    • cluster_points_radius()
    • cluster_objects_radius()
    • group_by_connectivity()
    • deduplicate_close_points()
    • pop_items()   <-- NEW (matches original popItems)
"""

from typing import List, Callable, Any, Tuple
import math
from collections import deque


# -------------------------------------------------------------------------
#  BASIC RADIUS CLUSTERING (point-based)
# -------------------------------------------------------------------------

def cluster_points_radius(points: List[Tuple[float, float]], radius: float) -> List[List[Tuple[float, float]]]:
    """
    Groups points that lie within a given Euclidean radius.
    """
    used = set()
    clusters = []

    for i, p in enumerate(points):
        if i in used:
            continue

        q = deque([i])
        cluster = []

        while q:
            idx = q.popleft()
            if idx in used:
                continue

            used.add(idx)
            cluster.append(points[idx])

            px, py = points[idx]
            for j, (ox, oy) in enumerate(points):
                if j in used:
                    continue
                if math.dist((px, py), (ox, oy)) <= radius:
                    q.append(j)

        clusters.append(cluster)

    return clusters


# -------------------------------------------------------------------------
#  OBJECT-BASED RADIUS CLUSTERING (e.g., Junction, Line)
# -------------------------------------------------------------------------

def cluster_objects_radius(objects: List[Any], radius: float, key: Callable[[Any], Tuple[float, float]]):
    """
    Generic object clustering using a coordinate extractor 'key'.
    Recreates the behavior of the BFS grouping.
    """
    coords = [key(o) for o in objects]
    idx_clusters = cluster_points_radius(coords, radius)

    clusters = []
    for cluster_coords in idx_clusters:
        sub = []
        for cx, cy in cluster_coords:
            for o in objects:
                if key(o) == (cx, cy):
                    sub.append(o)
                    break
        clusters.append(sub)

    return clusters


# -------------------------------------------------------------------------
#  CONNECTED COMPONENT GROUPING (BFS)
# -------------------------------------------------------------------------

def group_by_connectivity(items: List[Any], is_connected: Callable[[Any, Any], bool]):
    """
    Groups items based on a boolean connectivity rule.

    Used for:
        - grouping line segments into StraightEdges
        - grouping branch directions
    """
    visited = set()
    groups = []

    for i, obj in enumerate(items):
        if i in visited:
            continue

        queue = deque([i])
        comp = []

        while queue:
            idx = queue.popleft()
            if idx in visited:
                continue
            visited.add(idx)
            comp.append(items[idx])

            for j, other in enumerate(items):
                if j in visited:
                    continue
                if is_connected(items[idx], other):
                    queue.append(j)

        groups.append(comp)

    return groups


# -------------------------------------------------------------------------
#  POINT DEDUPLICATION
# -------------------------------------------------------------------------

def deduplicate_close_points(points: List[Tuple[float, float]], tolerance: float):
    """
    Deduplicates points that lie within `tolerance` distance of each other.
    """
    clusters = cluster_points_radius(points, tolerance)

    condensed = []
    for cluster in clusters:
        if len(cluster) == 1:
            condensed.append(cluster[0])
        else:
            xs = [p[0] for p in cluster]
            ys = [p[1] for p in cluster]
            condensed.append((sum(xs) / len(xs), sum(ys) / len(ys)))

    return condensed


# -------------------------------------------------------------------------
#  pop_items — IDENTICAL TO ORIGINAL popItems
# -------------------------------------------------------------------------

def pop_items(indices_to_pop: List[int], seq: List[Any]) -> List[Any]:
    """
    Removes all items whose indices appear in indices_to_pop.
    The indices are sorted in descending order so popping does not
    disturb the positions of items not yet removed.

    Example:
        pop_items([3,1], ['a','b','c','d']) -> ['a','c']

    Used heavily by:
        - junction_detector.merge_junctions()
        - junction classification pruning
    """
    if not indices_to_pop:
        return seq

    new_list = seq.copy()
    for idx in sorted(set(indices_to_pop), reverse=True):
        if 0 <= idx < len(new_list):
            new_list.pop(idx)
    return new_list
