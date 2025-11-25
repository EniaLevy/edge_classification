"""
Image I/O utilities for the junction detection pipeline.

This module provides:
    • load_images(path_pattern)
    • extract_numeric_id(filename)
    • ensure_output_dir(path)
    • save_image(path, image)

Handles all filesystem interaction in a consistent, testable way.
"""

import os
import re
import glob
from typing import List, Tuple

import cv2
import numpy as np


# -------------------------------------------------------------------------
#  FILENAME HANDLING
# -------------------------------------------------------------------------

def extract_numeric_id(filename: str) -> str:
    """
    Extract the first integer found in the filename.
    Used to keep naming consistent, matching the original script behavior.

    Example:
        'selected2/038.png' → '038'
    """
    m = re.search(r'\d+', filename)
    return m.group(0) if m else "0"


# -------------------------------------------------------------------------
#  IMAGE LOADING
# -------------------------------------------------------------------------

def load_images(path_pattern: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Loads all images matching the given glob pattern.

    Returns:
        images:  list of np.ndarray (BGR)
        names:   list of numeric identifiers extracted from filenames

    Example:
        images, names = load_images('selected2/*.png')
    """

    file_list = sorted(glob.glob(path_pattern))
    images = []
    names = []

    for fname in file_list:
        img = cv2.imread(fname)
        if img is None:
            continue
        images.append(img)
        names.append(extract_numeric_id(fname))

    return images, names


# -------------------------------------------------------------------------
#  OUTPUT DIRECTORY HANDLING
# -------------------------------------------------------------------------

def ensure_output_dir(path: str):
    """
    Ensures that an output directory exists.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# -------------------------------------------------------------------------
#  IMAGE SAVING
# -------------------------------------------------------------------------

def save_image(path: str, image: np.ndarray):
    """
    Save an image to disk, ensuring the directory exists.
    """
    ensure_output_dir(os.path.dirname(path))
    cv2.imwrite(path, image)
