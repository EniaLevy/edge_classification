import math
from typing import List, Tuple, Optional

import numpy as np
import cv2
import pybresenham as bres

from models.junction import Junction
from config import get_active_params


# ======================================================================
#  PUBLIC API
# ======================================================================

def extract_branches(
    junctions: List[Junction],
    image: np.ndarray,
    line_id_map: Optional[np.ndarray] = None,
    edges: Optional[np.ndarray] = None,
):
    """
    Parameters
    ----------
    junctions : list[Junction]
        Junction objects created and merged by junction_detector.
    image : np.ndarray
        BGR or grayscale image.
    line_id_map : np.ndarray, optional
        Map where each pixel stores the ID of the line segment it belongs to.
        If None, branch validation (tag check) still works but less strict.
    edges : np.ndarray, optional
        Canny edge map (uint8). If None, computed internally.

    Notes
    -----
    This function MUTATES each junction object:
        - calculateSearchRanges()
        - calculateCircumference()
        - checkJunctionCircumference()
        - turnInt()
        - addBranch(chain)
    """
    params = get_active_params()

    h, w = image.shape[:2]

    # -----------------------------------------------------------
    # Prepare edge map as in the original code
    # -----------------------------------------------------------
    if edges is None:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny_hi = params["CANNY_THRESHOLD"]
        canny_lo = params["CANNY_THRESHOLD_LOW"]
        edges = cv2.Canny(gray, canny_lo, canny_hi)
        edges = (edges > 0).astype(np.uint8) * 255  # convert to 0/255 as original

    # prepare line_id_map
    if line_id_map is None:
        line_id_map = np.zeros((h, w), dtype=np.int32)

    rad_thresh = params["RAD_THRESH"]      # 0.7 synthetic, 0.8 real
    tag_max = params["TAG_MAX"]            # 3 synthetic, 2 real

    # ==================================================================
    # Helper functions
    # ==================================================================

    def get_pixel_neighborhood(pt: Tuple[int, int], size: int):
        x0, y0 = pt
        pts = []
        for dx in range(-size, size + 1):
            for dy in range(-size, size + 1):
                pts.append((x0 + dx, y0 + dy))
        return pts

    def search_region(branch_pixels: List[Tuple[int, int]], tag_check=True):
        """
        Original search_region behavior:

            - For each branch point:
                * check neighborhood for an edge pixel
                * accumulate line_id tags near the ends of the branch
                * if the exact branch point is itself an edge pixel, append it
                immediately (first append)
                * if any edge was found in its neighborhood, append the branch
                point again (second append)

            - Abort and return [] if more than tag_max distinct tags appear.
        """
        chain = []
        tags = {0}

        for (x, y) in branch_pixels:
            try:
                if not tag_check:
                    neighborhood = get_pixel_neighborhood((x, y), size=2)
                else:
                    neighborhood = get_pixel_neighborhood((x, y), size=1)

                flag = False

                for nx, ny in neighborhood:
                    # bounds check
                    if nx < 0 or ny < 0 or nx >= w or ny >= h:
                        continue

                    if edges[ny, nx] > 0:
                        flag = True

                        # original: accumulate tags only near the branch ends
                        if tag_check and (x, y) not in branch_pixels[3:-3]:
                            tags.add(int(line_id_map[y, x]))

                        # original: if the edge pixel equals the branch point,
                        # append immediately
                        if nx == x and ny == y:
                            chain.append((x, y))
                        break

            except IndexError:
                flag = False

            # original: clear and abort if tags exceed tag_max
            if len(tags) > tag_max:
                return []

            # original: append a second time if any edge was found
            if flag:
                chain.append((x, y))

        return chain


    def branchez(j: Junction):
        """
        Full branch extraction for a single junction, matching the original
        'branchez' function from edge_classification_original.py:

          - calculate search ranges
          - compute outer circumference points near edges
          - compute inner junction circumference points near edges
          - for each outer point, connect to nearest inner point with
            a Bresenham line and run search_region
        """
        # 1) Search ranges
        j.calculateSearchRanges()

        # 2) Outer circumference based on search_ranges + range_threshold
        j.calculateCircumference()

        # --- Recreate ORIGINAL circumference logic using edges ---

        # Outer circle: keep points whose neighborhood hits an edge
        circle_pts = [
            (int(x), int(y))
            for x, y in bres.circle(round(j.x), round(j.y), round(j.r))
        ]
        circumference = []
        for cx, cy in circle_pts:
            for nx, ny in get_pixel_neighborhood((cx, cy), size=1):
                if 0 <= nx < w and 0 <= ny < h and edges[ny, nx] > 0:
                    circumference.append((nx, ny))
                    break
        # sort points around circle as in original
        if circumference:
            j.circumference = j.sortPoints(np.asarray(circumference))
        else:
            j.circumference = []

        # Inner junction circumference: radius = junction_radius
        j.junction_circumference = []
        inner_circle = [
            (int(x), int(y))
            for x, y in bres.circle(round(j.x), round(j.y), round(j.junction_radius))
        ]
        for px, py in inner_circle:
            if 0 <= px < w and 0 <= py < h and edges[py, px] > 0:
                j.junction_circumference.append((px, py))

        # 3) Round coordinates to ints
        j.turnInt()

        if not j.circumference or not j.junction_circumference:
            return

        # 4) For every point on outer circumference, build branch candidates
        for (cx, cy) in j.circumference:
            # find nearest inner junction point
            d_min = float("inf")
            nearest_j = None
            for (jx, jy) in j.junction_circumference:
                d = math.dist((cx, cy), (jx, jy))
                if d < d_min:
                    d_min = d
                    nearest_j = (jx, jy)

            if nearest_j is None:
                continue

            # Bresenham line: junction point -> outer circumference point
            branch_pixels = [
                (int(x), int(y))
                for x, y in bres.line(
                    nearest_j[0], nearest_j[1],
                    int(cx), int(cy)
                )
            ]

            # Validate region
            chain = search_region(branch_pixels)

            # Length threshold from original code
            min_len = rad_thresh * (j.r - j.junction_radius)
            if len(chain) > min_len:
                j.addBranch(chain)


    # ==================================================================
    # Execute branchez() for each junction
    # ==================================================================

    for j in junctions:
        try:
            branchez(j)
        except Exception:
            # keep processing pipeline robust
            continue

    return junctions
