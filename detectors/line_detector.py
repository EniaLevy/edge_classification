import cv2
import numpy as np

from models.line import Line
from utils.bresenham_utils import draw_line
from config import get_active_params, SYNTHETIC_MODE



def detect_lines(image_gray):
    """
    Detects line segments using LSD, wraps them as Line objects,
    and filters based on minimum length (radius_threshold/3).

    Parameters
    ----------
    image_gray : np.ndarray
        Grayscale input image.

    Returns
    -------
    list[Line]
        List of valid detected lines.
    """

    params = get_active_params()

    # Minimum allowable line length
    range_threshold = params["RANGE_THRESHOLD_PIXELS"]

    # ORIGINAL behavior:
    #   - synthetic: minimum_line_length = 0
    #   - real:      minimum_line_length = range_threshold / 3
    if SYNTHETIC_MODE:
        minimum_line_length = 0
    else:
        minimum_line_length = range_threshold * params["LINE_MIN_LENGTH_RATIO"]


    # LSD detector
    lsd = cv2.createLineSegmentDetector(0)
    detected = lsd.detect(image_gray)[0]

    if detected is None:
        return []

    # Reshape output to Nx2x2
    detected = detected.reshape(-1, 2, 2)

    lines = []
    for (pt1, pt2) in detected:
        line = Line([pt1, pt2])
        if line.length >= minimum_line_length:
            lines.append(line)

    return lines


def build_line_id_map(lines, shape_hw):
    """
    Builds a pixel-wise map of line IDs, exactly as in the original script.

    Every pixel that belongs to a line segment receives that line's ID.
    If multiple lines overlap, later IDs overwrite earlier ones—same behavior.

    Parameters
    ----------
    lines : list[Line]
        List of Line objects.
    shape_hw : tuple[int, int]
        Image height and width.

    Returns
    -------
    np.ndarray
        2D map where pixel values = line.id, or 0 if no line hit.
    """

    h, w = shape_hw
    line_id_map = np.zeros((h, w), dtype=np.int32)

    for line in lines:
        hits = draw_line(int(line.x1), int(line.y1), int(line.x2), int(line.y2))

        for x, y in hits:
            if 0 <= x < w and 0 <= y < h:
                line_id_map[y, x] = line.id

    return line_id_map
