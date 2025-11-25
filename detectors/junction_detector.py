import math
from typing import List, Tuple

import cv2
import numpy as np

from models.junction import Junction
from utils.geometry import (
    line_intersect,
    calculateAngle,
    checkIfParallel,
    getPixelNeighborhood,
)
from utils.clustering import pop_items, group_by_connectivity
from config import get_active_params


# ----------------------------------------------------------------------
# 1. INITIAL CANDIDATE DETECTION (line–line intersections)
# ----------------------------------------------------------------------

def detect_junction_candidates(lines: List, image) -> List[Junction]:
    """
    Detect raw junction candidates by checking all line–line intersections.

      - Rejects nearly parallel line pairs
      - Computes intersection point
      - Filters by distance-to-lines (belonging distance)
      - Expands to neighborhood pixels that lie on Canny edges
      - Marks exact-intersection pixel as centroid

    Parameters
    ----------
    lines : list[Line]
        Detected Line objects.
    image : np.ndarray
        Original BGR or grayscale image.

    Returns
    -------
    list[Junction]
        Raw junction candidates (with .line_segments filled).
    """
    params = get_active_params()
    distance_threshold = params["DISTANCE_THRESHOLD"]
    angle_threshold = params["ANGLE_THRESHOLD"]

    h, w = image.shape[:2]

    # ensure grayscale for Canny
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    canny_hi = params["CANNY_THRESHOLD"]
    canny_lo = params["CANNY_THRESHOLD_LOW"]
    edges = cv2.Canny(gray, canny_lo, canny_hi)

    candidates: List[Junction] = []

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):

            # angle filtering (skip nearly parallel pairs)
            ang = calculateAngle(lines[i], lines[j])
            if ang < angle_threshold or abs(ang - 180) < angle_threshold:
                continue

            # intersection point
            inter = line_intersect(lines[i].m, lines[i].b, lines[j].m, lines[j].b)
            if inter is None:
                continue

            x, y = inter
            if x <= 0 or x >= w or y <= 0 or y >= h:
                continue

            # base candidate at exact intersection
            base_junc = Junction([x, y])

            # if this intersection was already registered, just add lines to it
            if candidates and base_junc in candidates:
                existing = candidates[candidates.index(base_junc)]
                existing.addLineSeg(lines[i])
                existing.addLineSeg(lines[j])
                continue

            # belonging distances to both lines
            d1 = base_junc.calculateBelongingDistance(lines[i])
            d2 = base_junc.calculateBelongingDistance(lines[j])
            if d1 > distance_threshold or d2 > distance_threshold:
                continue

            # neighborhood expansion: every pixel around intersection that lies on an edge
            for nx, ny in getPixelNeighborhood((x, y), size=1):
                rx, ry = int(round(nx)), int(round(ny))

                # bounds check
                if rx < 0 or ry < 0 or rx >= w or ry >= h:
                    continue

                # must hit an edge pixel (Canny)
                if edges[ry, rx] == 0:
                    continue

                # centroid flag if position matches base intersection
                if math.isclose(rx, base_junc.x) and math.isclose(ry, base_junc.y):
                    jn = Junction([rx, ry], centr=True)
                else:
                    jn = Junction([rx, ry])

                # Add the two line segments involved in the intersection
                jn.addLineSeg(lines[i])
                jn.addLineSeg(lines[j])  # here j is the integer index from the outer loop ✓

                # Add candidate
                candidates.append(jn)

    return candidates


# ----------------------------------------------------------------------
# 2. MERGING RAW JUNCTION CANDIDATES
# ----------------------------------------------------------------------

def merge_junctions(candidates: List[Junction], image) -> List[Junction]:
    """
    Merge clusters of nearby junction candidates and compute
    centroid junctions, closely following BFS logic.

    Steps:
      - BFS cluster all junctions within DISTANCE_THRESHOLD/2
      - If cluster is small (<= 9) and has centroids:
            keep only centroids (radius = 2), drop others
      - Else:
            create a new Junction at the cluster centroid,
            radius = ceil(max distance to members),
            aggregate all line segments from cluster
      - Discard merged junctions whose junction_radius is larger than
        (r - junction_radius), where r is the outer search radius.
      - Finally, sort by original radius (og_r) and keep only a fraction
        defined by PERCENT_KEPT.
    """
    params = get_active_params()
    distance_threshold = params["DISTANCE_THRESHOLD"]
    percent_kept = params.get("PERCENT_KEPT", 1.0)

    visited: List[Junction] = []
    pop_indices: List[int] = []
    add: List[Junction] = []

    for i, junc in enumerate(candidates):
        if junc in visited:
            continue

        # BFS cluster of all junctions within distance_threshold/2
        search = [junc]
        cluster: List[Junction] = []

        while search:
            cur = search.pop(0)
            if cur in cluster:
                continue
            cluster.append(cur)
            visited.append(cur)

            for j in range(i + 1, len(candidates)):
                if candidates[j] in visited:
                    continue
                d = math.dist(cur.coordinates, candidates[j].coordinates)
                if d <= distance_threshold / 2:
                    search.append(candidates[j])

        # Small clusters (<=9) that contain centroids:
        #   - keep only centroid entries (junction_radius = 2)
        #   - drop others
        centroids = [c for c in cluster if c.centroid]
        if len(cluster) <= 9 and len(centroids) > 0:
            for c in centroids:
                c.junction_radius = 2
                # original also called branchez(c) here;
                # in this modular version branch extraction
                # will be done later by branch_finder.
            for c in cluster:
                if c not in centroids:
                    pop_indices.append(candidates.index(c))
            # keep centroids as-is, do not create a merged junction
            continue

        # Otherwise: aggregate entire cluster into a new centroid junction
        if not cluster:
            continue

        # unique line segments present in this cluster
        junction_lines = []
        for c in cluster:
            junction_lines.extend(c.line_segments)
        junction_lines = list(set(junction_lines))

        # collect positions & mark all cluster members for removal
        positions = []
        for c in cluster:
            positions.append(c.coordinates)
            pop_indices.append(candidates.index(c))

        positions = np.asarray(positions)
        mean_point = positions.mean(axis=0).tolist()
        new_junc = Junction(mean_point)

        # radius = ceil(max distance from cluster members to new_junc)
        dmax = 0.0
        for c in cluster:
            dis = math.dist(c.coordinates, new_junc.coordinates)
            if dis > dmax:
                dmax = dis
        new_junc.junction_radius = math.ceil(dmax)

        # combine line segments
        for line in junction_lines:
            new_junc.addLineSeg(line)

        # approximate original r-based filtering:
        #   original branchez() set r, then they kept only those
        #   with junction_radius <= (r - junction_radius)
        new_junc.calculateSearchRanges()
        new_junc.calculateCircumference()
        if new_junc.r > 0 and new_junc.junction_radius > (new_junc.r - new_junc.junction_radius):
            # discard junctions where inner radius is too large compared to search radius
            continue

        add.append(new_junc)

    # finalize: remove old members, add new ones
    merged = pop_items(pop_indices, candidates)
    merged.extend(add)

    # sort by original radius estimate (og_r) and keep top percent_kept
    if merged:
        merged.sort(
            key=lambda x: x.og_r if hasattr(x, "og_r") else x.junction_radius,
            reverse=True
        )
        k = int(len(merged) * percent_kept)
        merged = merged[:k]

    return merged


# ----------------------------------------------------------------------
# 3. CLASSIFY JUNCTIONS (L, T, Y, IY, P, X, K)
# ----------------------------------------------------------------------

def classify_junctions(junctions: List[Junction]) -> List[Junction]:
    """
    Final classification stage:

      - removes junctions with <2 branches
      - merges redundant branches within same direction cluster
      - discards junctions with <2 or >4 branches after reduction
      - assigns:
           .type  = "l", "t", "y", "iy", "x", "p", "k", or None
           .orientation = (x, y) for L/T/P/IY junctions
           .spine_branches / .branch_parallel_index as needed
    """

    # filter: keep only junctions that have at least 2 branches
    drop = []

    for idx, junc in enumerate(junctions):
        if len(junc.branches) < 2:
            drop.append(idx)
            continue

        # Ensure branches are sorted by direction as in original
        junc.sortBranches()

        # reduce redundant branches (keep longest in each angular cluster)
        reduce_branch_redundancy(junc)

        if len(junc.branches) < 2 or len(junc.branches) > 4:
            drop.append(idx)

    # remove invalid junctions
    junctions[:] = pop_items(drop, junctions)

    # classify each remaining junction
    for j in junctions:
        classify_single_junction(j)

    return junctions


# ----------------------------------------------------------------------
# 4. BRANCH REDUNDANCY REDUCTION
# ----------------------------------------------------------------------

def reduce_branch_redundancy(junction: Junction):
    """
    Keeps only the longest branch in each angular cluster, using a connectivity
    grouping that matches the original BFS-style logic:

      - branch i and j are connected if their directions differ by
        <= ANGLE_THRESHOLD2
      - connectivity is transitive (A~B, B~C → all in same group)
      - in each group, keep the branch with maximum pixel length
    """
    params = get_active_params()
    angle2_threshold = params["ANGLE_THRESHOLD2"]

    lines = junction.branch_lines
    if not lines:
        return

    # Build connectivity over indices 0..N-1
    indices = list(range(len(lines)))

    def is_connected(i, j):
        return abs(lines[i].direction - lines[j].direction) <= angle2_threshold

    idx_groups = group_by_connectivity(indices, is_connected)

    kept_indices = []
    for group in idx_groups:
        best = max(group, key=lambda idx: len(junction.branches[idx]))
        kept_indices.append(best)

    # rebuild branch and branch_lines arrays
    junction.branches = [junction.branches[k] for k in kept_indices]
    junction.branch_lines = [junction.branch_lines[k] for k in kept_indices]


# ----------------------------------------------------------------------
# 5. PER-JUNCTION CLASSIFICATION
# ----------------------------------------------------------------------

def classify_single_junction(j: Junction):
    """
    Sets:
        j.type ∈ {"l","t","y","iy","x","p","k",None}
        j.orientation = [x,y] for L/T/P/IY
        j.spine_branches + j.branch_parallel_index as needed
    """
    bcount = len(j.branches)

    # 2 branches → L or discarded if parallel
    if bcount == 2:
        if checkIfParallel(j.branch_lines[0], j.branch_lines[1]):
            j.type = None
        else:
            j.type = "l"
            set_orientation_L(j)
        return

    # 3 branches → T or Y
    if bcount == 3:
        parallel_count = count_parallel_branches(j)
        if parallel_count > 0:
            j.type = "t"
            set_orientation_T(j)
        else:
            j.type = "y"
            maybe_convert_to_inverse_y(j)
        return

    # 4 branches → P, X, or K
    if bcount == 4:
        parallel_count = count_parallel_branches(j)
        if parallel_count == 1:
            i, k = j.branch_parallel_index[0]
            # adjacent branches form degenerate K-junction
            if (i + 1 == k) or (i == 0 and k == 3):
                j.type = "k"
            else:
                j.type = "p"
                set_orientation_P(j)
        else:
            j.type = "x"
        return


# ----------------------------------------------------------------------
# 6. ORIENTATION HELPERS
# ----------------------------------------------------------------------

def set_orientation_L(j: Junction):
    """
    L-junction orientation is given by the sum of the two branch vectors.
    Result vector is extended to length j.r and translated back to center.
    """

    v = branch_to_vec(j.branches[0]) + branch_to_vec(j.branches[1])
    if np.linalg.norm(v) == 0:
        j.orientation = [j.x, j.y]
        return

    # unit vector scaled by junction outer radius (negative for direction)
    v = v / np.linalg.norm(v) * j.r * -1
    center = np.array([j.x, j.y])
    endpoint = (center + v).astype(int).tolist()
    j.orientation = endpoint


def set_orientation_T(j: Junction):
    """
    T-junction orientation is defined by the non-parallel branch.
    """
    if not j.branch_parallel_index:
        return

    parallel_pair = j.branch_parallel_index[0]
    for idx, branch in enumerate(j.branches):
        if idx not in parallel_pair:
            j.orientation = branch[-1]
            return


def set_orientation_P(j: Junction):
    """
    P-junction orientation depends on the angle between the two parallel branches.
    """
    if not j.branch_parallel_index:
        return

    i = j.branch_parallel_index[0][0]
    # choose orientation from parallel branch, depending on direction gap
    if j.branch_lines[i + 1].direction - j.branch_lines[i].direction < 90:
        j.orientation = j.branches[i][-1]
    else:
        j.orientation = j.branches[j.branch_parallel_index[0][1]][-1]


def maybe_convert_to_inverse_y(j: Junction):
    """
    For a 3-branch Y junction, check if two branches lie within 90°
    of one another. If so, treat as inverse-Y ("iy") and set orientation.
    """
    for i in range(len(j.branch_lines)):
        count = 0
        for k in range(len(j.branch_lines)):
            if k == i:
                continue
            if abs(j.branch_lines[i].direction - j.branch_lines[k].direction) < 90:
                count += 1
                if count >= 2:
                    j.type = "iy"
                    j.orientation = j.branches[i][-1]
                    j.spine_branches.append(i)
                    return


# ----------------------------------------------------------------------
# 7. PARALLEL COUNTS
# ----------------------------------------------------------------------

def count_parallel_branches(j: Junction) -> int:
    """
    Counts parallel branch pairs for a junction and fills:
        j.branch_parallel_index
        j.spine_branches   (for the simple case where exactly one pair is found)
    """
    j.branch_parallel_index.clear()
    j.spine_branches = []

    bl = j.branch_lines
    count = 0

    for i in range(len(bl)):
        for k in range(i + 1, len(bl)):
            if checkIfParallel(bl[i], bl[k]):
                count += 1
                j.branch_parallel_index.append([i, k])

    if count == 1:
        j.spine_branches.extend(j.branch_parallel_index[0])

    return count


# ----------------------------------------------------------------------
# 8. VECTOR UTILITY
# ----------------------------------------------------------------------

def branch_to_vec(branch: List[Tuple[int, int]]):
    """
    Convert a branch (sequence of pixels) into a vector from
    last point to first point, matching the original convention.
    """

    a1 = np.array(branch[-1])
    a2 = np.array(branch[0])
    return a2 - a1
