import cv2
import numpy as np

from utils.image_io import load_images, ensure_output_dir
from detectors.line_detector import detect_lines, build_line_id_map
from detectors.junction_detector import (
    detect_junction_candidates,
    merge_junctions,
    classify_junctions,
)
from detectors.branch_finder import extract_branches
from detectors.edge_classifier import (
    group_into_straight_edges,
    classify_edge_objects,
    build_predicted_map,
)

from visualization.save_outputs import save_all_outputs

from config import (
    SELECTED_IMAGE_PATTERN,
    OUTPUT_FOLDER,
    get_active_params,
)


def process_image(image, image_name: str):
    """
    Runs the complete pipeline for one image:
      1. Edge detection (Canny)
      2. Line detection (LSD)
      3. Junction detection & merging
      4. Branch extraction
      5. Junction classification
      6. Straight-edge grouping
      7. Edge-object classification
      8. Prediction map & line-id map processing
      9. Save all outputs (predictions, junctions, edges, linemap, classified)
    """

    print(f"\n=== Processing image with name: {image_name} ===")
    params = get_active_params()

    # ------------------------------
    # STEP 0 — PREP: GRAYSCALE & EDGES
    # ------------------------------
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    canny_hi = params["CANNY_THRESHOLD"]
    canny_lo = params["CANNY_THRESHOLD_LOW"]
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # ------------------------------
    # STEP 1 — LINE DETECTION
    # ------------------------------
    lines = detect_lines(gray)
    if not lines:
        print(f"[WARN] No lines detected in {image_name}. Skipping.")
        return

    line_id_map = build_line_id_map(lines, gray.shape[:2])

    # ------------------------------
    # STEP 2 — JUNCTION CANDIDATES
    # ------------------------------
    candidates = detect_junction_candidates(lines, image)

    # ------------------------------
    # STEP 3 — MERGE & FILTER JUNCTIONS
    # ------------------------------
    junctions = merge_junctions(candidates, image)

    # ------------------------------
    # STEP 4 — BRANCH EXTRACTION
    # (uses same edges & line_id_map as above)
    # ------------------------------
    extract_branches(junctions, image, line_id_map=line_id_map, edges=edges)

    # ------------------------------
    # STEP 5 — CLASSIFY JUNCTIONS
    # ------------------------------
    classify_junctions(junctions)

    # ------------------------------
    # STEP 6 — GROUP STRAIGHT EDGES
    # ------------------------------
    straight_edges = group_into_straight_edges(lines)

    # ------------------------------
    # STEP 7 — CLASSIFY EDGE OBJECTS
    # ------------------------------
    classify_edge_objects(straight_edges, junctions)

    # ------------------------------
    # STEP 8 — BUILD PREDICTION MAP
    # ------------------------------
    predicted_map = build_predicted_map(straight_edges, gray.shape[:2])

    # Process line_id_map as in original script:
    # shift IDs into [0,255] and then binarize (0 vs 255)
    max_id = np.amax(line_id_map) if line_id_map.size > 0 else 0
    dif = round(255 - max_id)
    if dif < 0:
        dif = 0
    line_id_map_u8 = (line_id_map + dif).astype(np.uint8)
    line_id_map_u8[line_id_map_u8 == dif] = 0
    line_id_map_u8[line_id_map_u8 > 0] = 255

    # ------------------------------
    # STEP 9 — SAVE OUTPUTS
    # ------------------------------
    save_all_outputs(
        output_dir=OUTPUT_FOLDER,
        image_id=image_name,
        base_image=image,
        edges_bgr=edges_bgr,
        predicted_map=predicted_map,
        line_id_map=line_id_map_u8,
        junctions=junctions,
        straight_edges=straight_edges,
    )

    print(f"[OK] Finished {image_name}")


def main():
    """
    Main entry point:
      - Loads images
      - Processes each one independently
      - Saves output files
    """
    ensure_output_dir(OUTPUT_FOLDER)

    images, names = load_images(SELECTED_IMAGE_PATTERN)
    if not images:
        print(f"[ERROR] No images matched pattern: {SELECTED_IMAGE_PATTERN}")
        return

    for img, name in zip(images, names):
        process_image(img, name)

    print("\n=== All images processed ===")


if __name__ == "__main__":
    main()
