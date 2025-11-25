import math
from typing import List, Tuple

import numpy as np

from models.line import Line
from utils.bresenham_utils import bres_circle
from config import get_active_params


class Junction:
    """
    It supports:
      - association with multiple line segments
      - search radius computation for branch extraction
      - circumference generation
      - branch storage and branch-line proxies
      - classification fields (type, orientation, spine branches, etc.)

    Notes:
      • Coordinates are stored both as `coordinates` (original name) and `coords`
        to remain compatible with the refactored detectors.
      • Branch lines are real Line objects (for direction + parallel checks).
    """

    junct_id = 0

    def __init__(self, coords: List[float], centr: bool = False):
        Junction.junct_id += 1
        self.id = Junction.junct_id

        # coordinate storage (both names for compatibility)
        self.coordinates: List[float] = coords
        self.coords: List[float] = coords

        self.x: float = coords[0]
        self.y: float = coords[1]
        self.centroid: bool = centr  # legacy flag from original code

        # radii
        self.junction_radius: int = 1
        self.og_r: float = 0
        self.r: float = 0

        # circles
        self.junction_circumference: List[Tuple[int, int]] = []
        self.circumference: List[Tuple[int, int]] = []

        # branch & line associations
        self.branches: List[List[Tuple[int, int]]] = []
        self.branch_lines: List[Line] = []
        self.branch_parallel_index: List[List[int]] = []
        self.spine_branches: List[int] = []

        self.line_segments: List[Line] = []
        self.line_distances: List[float] = []

        self.search_ranges: List[int] = []

        # classification-related fields
        self.type = None
        self.belonging_distance = None
        self.orientation = None

        # flags used by clustering logic
        self.is_centroid: bool = False

    # ------------------------------------------------------------------
    # Equality & hashing (by position, as in original)
    # ------------------------------------------------------------------

    def __eq__(self, other):
        return (
            isinstance(other, Junction)
            and math.isclose(self.x, other.x)
            and math.isclose(self.y, other.y)
        )

    def __hash__(self):
        return hash(("coordinates", tuple(self.coordinates)))

    # ------------------------------------------------------------------
    # Line associations
    # ------------------------------------------------------------------

    def addLineSeg(self, line_segment: Line):
        """
        Add a line segment associated to this junction (can be more than two).
        """
        if line_segment not in self.line_segments:
            self.line_segments.append(line_segment)
            self.line_distances.append(self.calculateBelongingDistance(line_segment))

    # alias used by refactored detectors
    def add_line(self, line_segment: Line):
        self.addLineSeg(line_segment)

    # ------------------------------------------------------------------
    # Branch management
    # ------------------------------------------------------------------

    def addBranch(self, branch: List[Tuple[int, int]]):
        """
        Add a branch (list of pixel coordinates) and create its Line proxy.
        The branch is only added if not already present.
        """
        if branch not in self.branches and len(branch) >= 2:
            self.branches.append(branch)
            self.addBranchLine([branch[0], branch[-1]])

    def addBranchLine(self, branch_edges: List[Tuple[int, int]]):
        """
        Create a Line object from branch endpoints and store it.
        Direction is computed in Line.__init__, as in the original when
        calculate_direction=True was passed.
        """
        line = Line(branch_edges)
        self.branch_lines.append(line)

    def sortBranches(self):
        """
        Sort branches and branch_lines by line direction, as in original:

            self.branches reordered in the same order as self.branch_lines
            when branch_lines are sorted by .direction
        """
        if not self.branch_lines:
            return

        # zip(branch_line, branch), sort by line.direction, then unzip
        pairs = list(zip(self.branch_lines, self.branches))
        pairs.sort(key=lambda x: x[0].direction)

        self.branch_lines = [p[0] for p in pairs]
        self.branches = [p[1] for p in pairs]

    # ------------------------------------------------------------------
    # Search range calculation (for branch extraction)
    # ------------------------------------------------------------------

    def calculateSearchRanges(self):
        """
        Compute candidate search radii per line segment.

        This is a direct adaptation of the original:

            if junction lies on line:
                r = distance to line (clamped by distance_threshold)
            else:
                r = distance to closest point on line; if < threshold,
                    use full line length as radius.

        Distances are stored in self.search_ranges.
        """
        params = get_active_params()
        distance_threshold = params["DISTANCE_THRESHOLD"]

        self.search_ranges = []

        for line in self.line_segments:
            r = 0
            if line.pointIsOnLine(self.coordinates):
                r = line.distanceFromPointToLineSegment(self.coordinates)
                r = max(distance_threshold, r)
            else:
                closest_point = line.closestPointOnLine(self.coordinates)
                r = line.distanceFromEdgepoints(closest_point)
                if r < distance_threshold:
                    r = line.length
            r = int(r)
            self.search_ranges.append(r)

    # ------------------------------------------------------------------
    # Circumference generation
    # ------------------------------------------------------------------

    def calculateCircumference(self):
        """
        Compute outer search radius (self.r) and generate circumference points.

        Original logic:

            self.og_r = min(self.search_ranges)
            radius = min(self.og_r, range_threshold) + self.junction_radius
        """
        params = get_active_params()
        range_threshold = params["RANGE_THRESHOLD_PIXELS"]

        if self.search_ranges:
            self.og_r = min(self.search_ranges)
        else:
            # fallback if not set (rare)
            self.og_r = range_threshold

        radius = min(self.og_r, range_threshold)
        radius = radius + self.junction_radius
        self.r = radius

        # Bresenham circle centered at (x,y)
        circle = bres_circle(round(self.x), round(self.y), round(self.r))
        self.circumference = self.sortPoints(np.asarray(circle)) if circle else []

    def checkJunctionCircumference(self):
        """
        Generate the small inner circle around the junction center.
        """
        circle = bres_circle(round(self.x), round(self.y), round(self.junction_radius))
        self.junction_circumference = list(circle) if circle else []

    def sortPoints(self, xy: np.ndarray) -> List[Tuple[int, int]]:
        """
        Sort 2D points around the center by angle, preserving original behavior:

          - normalize x,y to [-1,1]
          - compute atan2(x_norm, y_norm)
          - sort by angle
        """
        if xy.size == 0:
            return []

        xy_sort = np.empty_like(xy, dtype=float)

        # normalize each axis to [-1,1], guarding against zero range
        for dim in (0, 1):
            col = xy[:, dim].astype(float)
            c_min = np.min(col)
            c_max = np.max(col)
            denom = (c_max - c_min)
            if denom == 0:
                xy_sort[:, dim] = 0.0
            else:
                xy_sort[:, dim] = 2.0 * (col - c_min) / denom - 1.0

        sort_array = np.arctan2(xy_sort[:, 0], xy_sort[:, 1])
        sort_result = np.argsort(sort_array)

        sorted_xy = xy[sort_result]
        return [tuple(p) for p in sorted_xy]

    # ------------------------------------------------------------------
    # Distances
    # ------------------------------------------------------------------

    def getLinesAndDistance(self):
        """
        Return list of (line_segment, belonging_distance) pairs.
        """
        return list(zip(self.line_segments, self.line_distances))

    def calculateBelongingDistance(self, line: Line) -> float:
        """
        Compute distance from this junction to a line segment:

          - 0 if point lies on the line (infinite line + segment check)
          - perpendicular distance to closest point on segment otherwise
        """
        distance = 0.0
        if not line.pointIsOnLine(self.coordinates):
            distance = line.distanceFromPointToLineSegment(self.coordinates)
        return distance

    # ------------------------------------------------------------------
    # Integer coordinate conversion
    # ------------------------------------------------------------------

    def turnInt(self):
        """
        Round the floating-point coordinates to nearest pixel and update internal
        state accordingly.
        """
        self.coordinates = [round(self.x), round(self.y)]
        self.coords = self.coordinates
        self.x = self.coordinates[0]
        self.y = self.coordinates[1]

    # ------------------------------------------------------------------
    # Centroid flag (used by merge_junctions)
    # ------------------------------------------------------------------

    def setAsCentroid(self):
        self.is_centroid = True

    # ------------------------------------------------------------------
    # Convenience / debugging
    # ------------------------------------------------------------------

    def __repr__(self):
        return f"Junction(coords={self.coordinates}, type={self.type}, r={self.r})"
