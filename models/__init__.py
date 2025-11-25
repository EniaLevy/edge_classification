"""
Data Models

Defines the core data structures:
- Line
- Junction
- StraightEdge
"""

from .line import Line
from .junction import Junction
from .straight_edge import StraightEdge

__all__ = ["Line", "Junction", "StraightEdge"]
