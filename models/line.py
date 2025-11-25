import math


class Line:
    """
    Supports:
      - slope / intercept computation
      - robust vertical-line surrogate slope
      - projection of points on infinite line
      - distance from point to *segment* (as in original script)
      - point-on-line tests with exact vs distance-threshold modes
      - 0–180° direction computation
    """

    id_num = 0

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def __init__(self, edgepoints):
        """
        edgepoints: [[x1, y1], [x2, y2]]
        """
        Line.id_num += 1
        self.id = Line.id_num

        self.edge_points = edgepoints
        self.a1 = edgepoints[0]
        self.a2 = edgepoints[1]

        self.x1, self.y1 = self.a1
        self.x2, self.y2 = self.a2

        # slope & intercept
        self.m = self.slope()
        self.b = self.y_intercept()

        # length of segment
        self.length = self.lineLength()

        # direction angle (0–180 degrees)
        self.direction = self.calculateDirection()

    # ------------------------------------------------------------
    # Basic geometric properties
    # ------------------------------------------------------------
    def slope(self):
        """
        Vertical lines in the original script returned a very large number
        (not math.inf) to avoid division issues.
        """
        if self.x1 == self.x2:  # vertical line surrogate
            return 99999999
        return (self.y2 - self.y1) / (self.x2 - self.x1)

    def y_intercept(self):
        """b = y - m x"""
        return self.y1 - self.m * self.x1

    def lineLength(self):
        return math.dist(self.a1, self.a2)

    # ------------------------------------------------------------
    # Direction logic
    # ------------------------------------------------------------
    def angle_trunc(self, a):
        """
        Original implementation truncated into [0, 2π] by looping.
        """
        while a < 0.0:
            a += math.pi * 2
        while a > (math.pi * 2):
            a -= math.pi * 2
        return a

    def calculateDirection(self):
        """
        direction = degrees(atan2(dy, dx)), truncated into [0, 360) as in the
        original thesis script (no modulo 180).
        """
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        angle = math.degrees(self.angle_trunc(math.atan2(dy, dx)))
        # Keep full [0, 360) range, do NOT mod 180
        return angle


    # ------------------------------------------------------------
    # Distance & projection
    # ------------------------------------------------------------
    def distanceFromPoint(self, point):
        """
        Distance from point to infinite line.

        Uses ax + by + c = 0 implicit line form for robust distance.
        """
        x0, y0 = point

        # handle vertical surrogate
        if self.m == 99999999:
            return abs(x0 - self.x1)

        a = -self.m
        b = 1
        c = -self.b
        return abs(a * x0 + b * y0 + c) / math.sqrt(a * a + b * b)

    def closestPointOnLine(self, p3):
        """
        Projection of p3 onto the infinite line defined by endpoints.
        Equivalent to original 'closestPointOnLine'.

        Returns projected point even if outside the segment.
        """
        (x1, y1), (x2, y2), (x3, y3) = self.a1, self.a2, p3
        dx, dy = (x2 - x1), (y2 - y1)
        det = dx * dx + dy * dy

        if det == 0:  # degenerate line
            return (x1, y1)

        a = (dy * (y3 - y1) + dx * (x3 - x1)) / det
        return (x1 + a * dx, y1 + a * dy)

    def distanceFromEdgepoints(self, point):
        """
        Fallback distance used in original code:
          min(distance to endpoint1, distance to endpoint2)
        """
        return min(math.dist(point, self.a1), math.dist(point, self.a2))

    def distanceFromPointToLineSegment(self, point):
        """
        1. Compute closest point on infinite line
        2. If projection lies ON SEGMENT → use perpendicular distance
        3. Else → use min distance to endpoints
        """
        p4 = self.closestPointOnLine(point)

        if self.pointIsOnSegment(p4):
            return math.dist(point, p4)
        else:
            return self.distanceFromEdgepoints(point)

    # ------------------------------------------------------------
    # Segment membership tests
    # ------------------------------------------------------------
    def pointIsOnSegment(self, point):
        """
        Bounding-box segment check identical to original method.
        """
        x0, y0 = point
        if not (min(self.x1, self.x2) <= x0 <= max(self.x1, self.x2)):
            return False
        if not (min(self.y1, self.y2) <= y0 <= max(self.y1, self.y2)):
            return False
        return True

    def pointIsOnLine(self, point, close=False, segment=True, dis_thres=0):
        """
        • close=False:
              exact line equation check (y == m x + b)
        • close=True:
              distanceFromPoint <= dis_thres

        • segment=True:
              must also lie within segment bounding box
        """
        x, y = point

        if not close:
            # exact match to line equation
            condition = math.isclose(
                y, (self.m * x + self.b),
                rel_tol=0.0,
                abs_tol=1e-4
            )
        else:
            dist = self.distanceFromPoint(point)
            condition = dist <= dis_thres

        if condition:
            if segment:
                return self.pointIsOnSegment(point)
            return True

        return False

    # ------------------------------------------------------------
    # Small helpers for external API compatibility
    # ------------------------------------------------------------
    def distance_from_point(self, x, y):
        """Wrapper for compatibility with junction_detector."""
        return self.distanceFromPoint((x, y))

    # ------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------
    def __repr__(self):
        return f"Line(id={self.id}, a1={self.a1}, a2={self.a2}, dir={self.direction:.1f})"
