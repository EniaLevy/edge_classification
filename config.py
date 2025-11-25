"""
Configuration file for the junction-detection system.

Contains both REAL and SYNTHETIC parameter sets as in the original script.
Modules should read values using the get_active_params() function.
"""

# ---------------------------------------------------------------
# MODE SELECTION
# ---------------------------------------------------------------

# Set to True when using digitally-created images
SYNTHETIC_MODE = True


# ---------------------------------------------------------------
# I/O PATHS
# ---------------------------------------------------------------

SELECTED_IMAGE_PATTERN = "selected/*.png"
OUTPUT_FOLDER = "output"


# ===============================================================
# REAL-MODE PARAMETERS
# ===============================================================

REAL = {
    "CANNY_THRESHOLD": 255,
    "RAD_THRESH": 0.8,
    "PERCENT_KEPT": 0.8,
    "TAG_MAX": 2
}


# ===============================================================
# SYNTHETIC-MODE PARAMETERS
# ===============================================================

SYNTH = {
    "CANNY_THRESHOLD": 90,
    "RAD_THRESH": 0.7,
    "PERCENT_KEPT": 1.0,
    "TAG_MAX": 3
}


# ---------------------------------------------------------------
# SHARED PARAMETERS (used in both modes)
# ---------------------------------------------------------------

DISTANCE_THRESHOLD = 10            # Td
ANGLE_THRESHOLD = 15               # branch parallel threshold
ANGLE_THRESHOLD2 = ANGLE_THRESHOLD * 1.5
RANGE_THRESHOLD_PIXELS = 30        # r value (before scaling)
LINE_MIN_LENGTH_RATIO = 1 / 3      # = range_threshold / 3

NEIGHBORHOOD_SIZE = 1
NEIGHBORHOOD_SIZE_LARGE = 2


# ---------------------------------------------------------------
# EDGE CLASSIFICATION SCORING WEIGHTS
# ---------------------------------------------------------------

EDGE_SCORES = {
    "L_OB": 1.0,
    "L_SC": 0.8,
    "T_OB": 1.5,
    "TB_OB": 0.8,
    "TB_RC": 0.8,
    "PS_SC": 1.0,
    "PB_RC": 1.0,
    "Y_SC": 1.3,
    "YS_SC": 1.0,
    "YB_OB": 1.0,
    "X_RC": 0.8
}

EDGE_OB_THRESHOLD = 0.1


# ---------------------------------------------------------------
# VISUALIZATION COLORS
# ---------------------------------------------------------------

COLOR_OB = (255, 0, 0) #Obstruction edges - blue
COLOR_SC = (0, 255, 0) #Surface edges - green
COLOR_RC = (0, 0, 255) #Reflectance edges - red
COLOR_OTHER = (255, 255, 255) #Unclassified edges - white


# ---------------------------------------------------------------
# PARAMETER ACCESS LOGIC
# ---------------------------------------------------------------

def get_active_params():
    """
    Returns the active set of parameters:
    - A combination of SHARED + mode-specific constants.
    - Used by detectors and classifiers so they only import one dictionary.
    """

    base = {
        "DISTANCE_THRESHOLD": DISTANCE_THRESHOLD,
        "ANGLE_THRESHOLD": ANGLE_THRESHOLD,
        "ANGLE_THRESHOLD2": ANGLE_THRESHOLD2,
        "RANGE_THRESHOLD_PIXELS": RANGE_THRESHOLD_PIXELS,
        "LINE_MIN_LENGTH_RATIO": LINE_MIN_LENGTH_RATIO,
        "NEIGHBORHOOD_SIZE": NEIGHBORHOOD_SIZE,
        "NEIGHBORHOOD_SIZE_LARGE": NEIGHBORHOOD_SIZE_LARGE,
        "EDGE_SCORES": EDGE_SCORES,
        "EDGE_OB_THRESHOLD": EDGE_OB_THRESHOLD
    }

    # Merge in real or synthetic mode values
    if SYNTHETIC_MODE:
        base.update(SYNTH)
    else:
        base.update(REAL)

    # Compute low threshold dynamically
    base["CANNY_THRESHOLD_LOW"] = base["CANNY_THRESHOLD"] / 3

    return base
