import math
from dataclasses import dataclass, field
from typing import List, Optional

from models.line import Line
from models.junction import Junction


@dataclass
class StraightEdge:
    """
    Represents a connected group of nearly-parallel line segments that form
    a larger straight edge structure in the image.

    Handles:
      • grouping line segments
      • collecting junctions that lie on the edge
      • classifying edge type (ob / sc / rc)
      • attaching branch type labels (ts, tb, ps, pb, ys, yb)
    """

    line_segments: List[Line] = field(default_factory=list)
    junctions: List[Junction] = field(default_factory=list)
    junction_types: List[str] = field(default_factory=list)
    branch_types: List[str] = field(default_factory=list)

    # Scoring weights (from original script)
    ob_score: float = 0.02
    rc_score: float = 0.0
    sc_score: float = 0.01

    type: Optional[str] = None  # ob, sc, rc

    # -------------------------------------------------------------
    #   Add items
    # -------------------------------------------------------------

    def addLineSeg(self, seg: Line):
        if seg not in self.line_segments:
            self.line_segments.append(seg)

    def addJunction(self, j: Junction):
        if j not in self.junctions:
            self.junctions.append(j)
            if j.type is not None:
                self.junction_types.append(j.type)

    # -------------------------------------------------------------
    #   Scoring & type classification
    # -------------------------------------------------------------

    def calculateScore(self):
        """
        Reproduces weight accumulation from original script:

            L junction → ob+, sc+
            T small → ob++ (ts)
            T big → ob+, rc+
            P small → sc+
            P big → rc+
            Y → sc++
            Y small → sc+
            Y big → ob+
            X → rc+ (we kept original behavior)
        """

        types = self.junction_types + self.branch_types

        for t in types:
            match t:
                case "l":
                    self.ob_score += 1
                    self.sc_score += 0.8

                case "ts":
                    self.ob_score += 1.5

                case "tb":
                    self.ob_score += 0.8
                    self.rc_score += 0.8

                case "ps":
                    self.sc_score += 1

                case "pb":
                    self.rc_score += 1

                case "y":
                    self.sc_score += 1.3

                case "ys":
                    self.sc_score += 1

                case "yb":
                    self.ob_score += 1

                case "x":
                    self.rc_score += 0.8

    def checkType(self):
        """
        Sets final type:
            ob = "outer boundary"
            sc = "side chain"
            rc = "ridge connection"
        """

        if self.ob_score > self.rc_score and self.ob_score > self.sc_score:
            if self.ob_score > 0.1:
                self.type = "ob"

        elif self.rc_score > self.ob_score and self.rc_score > self.sc_score:
            self.type = "rc"

        elif self.sc_score > self.ob_score and self.sc_score > self.rc_score:
            self.type = "sc"

