"""
Centralized output-saving utilities for the junction detection pipeline.

This module provides:
    • save_all_outputs(...)
    • save_predicted_map(...)
    • save_junctions(...)
    • save_edges(...)
    • save_line_id_map(...)
    • save_image(...)

Uses draw modules to visualize and utils.image_io for filesystem handling.
"""

import numpy as np
import cv2
from typing import List

from models.line import Line
from models.junction import Junction
from models.straight_edge import StraightEdge

from visualization.draw_lines import draw_lines, draw_colored_edges
from visualization.draw_junctions import draw_junctions
from utils.image_io import save_image, ensure_output_dir


# -------------------------------------------------------------------------
#   Save individual components
# -------------------------------------------------------------------------

def save_predicted_map(path: str, predicted: np.ndarray):
    """
    Saves the predicted class label map (uint8).
    """
    save_image(path, predicted)


def save_junctions(path: str, base_image: np.ndarray, junctions: List[Junction]):
    """
    Draw junctions on a copy of the base image and save to disk.
    """
    vis = base_image.copy()
    draw_junctions(vis, junctions)
    save_image(path, vis)


def save_edges(path: str, edges: np.ndarray):
    """
    Saves the grayscale or BGR edge image to disk.
    Matches original monolithic behavior.
    """
    save_image(path, edges)


def save_line_id_map(path: str, line_id_map: np.ndarray):
    """
    Writes the binary or labeled line_id_map to disk.
    """
    # Ensure uint8 output
    if line_id_map.dtype != np.uint8:
        line_id_map = line_id_map.astype(np.uint8)
    save_image(path, line_id_map)


def save_classified_edges(path: str, base_image: np.ndarray, straight_edges: List[StraightEdge]):
    """
    Draws the classified StraightEdge objects and saves the result.
    """
    vis = base_image.copy()
    draw_colored_edges(vis, straight_edges)
    save_image(path, vis)


# -------------------------------------------------------------------------
#   Master save function (used by main.py)
# -------------------------------------------------------------------------

def save_all_outputs(
    output_dir: str,
    image_id: str,
    base_image: np.ndarray,
    edges_bgr: np.ndarray,
    predicted_map: np.ndarray,
    line_id_map: np.ndarray,
    junctions: List[Junction],
    straight_edges: List[StraightEdge]
):
    """
    Saves every output artifact for one processed image.
    Matches the naming convention from the original monolithic script.

    Example output:
        <id>_predictions.png
        <id>_junctions.png
        <id>_edges.png
        <id>_linemap.png
        <id>_classified.png
    """

    ensure_output_dir(output_dir)

    # 1) Predicted map (rc/sc/ob labels)
    save_predicted_map(
        f"{output_dir}/{image_id}_predictions.png",
        predicted_map
    )

    # 2) Junction visualization
    save_junctions(
        f"{output_dir}/{image_id}_junctions.png",
        base_image,
        junctions
    )

    # 3) Edge image
    save_edges(
        f"{output_dir}/{image_id}_edges.png",
        edges_bgr
    )

    # 4) Line-ID map
    save_line_id_map(
        f"{output_dir}/{image_id}_linemap.png",
        line_id_map
    )

    # 5) Classified straight edges
    save_classified_edges(
        f"{output_dir}/{image_id}_classified.png",
        base_image,
        straight_edges
    )
