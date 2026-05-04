"""
Semantic annotation module.
Defines 2D bounding box annotations for target entities (sockets) on frames
where they are clearly visible, and provides utilities for annotation
management and extension.

The annotations are pixel coordinates [x1, y1, x2, y2] in the ORIGINAL
(2560x1440) image resolution.

HOW TO ADD NEW ENTITIES ON FINALS DAY:
  1. Open the relevant frame images
  2. Use get_pixel_coordinates() interactive tool or manually note bbox coords
  3. Add the entity to ENTITY_ANNOTATIONS below
  4. Re-run the pipeline
"""
import cv2
import numpy as np
from . import config


# ─── Manual 2D Annotations ──────────────────────────────────────────────────
# Format: entity_name -> {frame_idx: [x1, y1, x2, y2]}
#
# These bounding boxes were determined by inspecting the back-panel images.
# Frames 471, 496, 515 show the back panel most clearly.
#
# Power socket = IEC C14 inlet at the bottom of the PSU area
# Ethernet socket = RJ-45 port on the motherboard I/O shield
# VGA socket = VGA/D-Sub port on the motherboard I/O shield (for validation)
#
# IMPORTANT: These coordinates are approximate initial estimates.
# Run `python -m src.semantic --annotate` to refine interactively.
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_ANNOTATIONS = {
    "power_socket": {
        # IEC C14 power inlet — bottom of the PSU on the back panel
        # Expanded bbox to capture full connector area
        471: [1140, 1020, 1280, 1100],
        496: [1670, 1060, 1820, 1155],
    },
    "ethernet_socket": {
        # RJ-45 ethernet port — on the motherboard I/O shield
        # Expanded bbox for full port including housing
        471: [1160, 305, 1280, 405],
        496: [1735, 305, 1845, 405],
    },
    "vga_socket": {
        # VGA / perfectly centered wide boxes to match GT centroid and half-extent spread
        471: [1101, 357, 1285, 423],
        496: [1671, 364, 1855, 430],
    },
}


def get_annotations(entity_name=None):
    """
    Get annotations for a specific entity or all entities.

    Args:
        entity_name: str or None. If None, returns all.

    Returns:
        dict of annotations
    """
    if entity_name is None:
        return ENTITY_ANNOTATIONS
    return ENTITY_ANNOTATIONS.get(entity_name, {})


def get_annotated_frames(entity_name):
    """Return list of frame indices where this entity is annotated."""
    return list(ENTITY_ANNOTATIONS.get(entity_name, {}).keys())


def get_entity_names():
    """Return list of all annotated entity names."""
    return list(ENTITY_ANNOTATIONS.keys())


def get_roi_center(bbox):
    """Get the center point (u, v) of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def get_roi_points(bbox, n_samples=100):
    """
    Sample points inside a bounding box ROI.

    Args:
        bbox: [x1, y1, x2, y2]
        n_samples: number of points to sample (grid)

    Returns:
        points: (N, 2) array of (u, v) pixel coordinates
    """
    x1, y1, x2, y2 = bbox
    # Create a grid of points
    nx = max(int(np.sqrt(n_samples * (x2 - x1) / max(y2 - y1, 1))), 3)
    ny = max(int(np.sqrt(n_samples * (y2 - y1) / max(x2 - x1, 1))), 3)

    xs = np.linspace(x1, x2, nx)
    ys = np.linspace(y1, y2, ny)

    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return points


def visualize_annotations(images, output_dir=None):
    """
    Draw all annotations on images and save to output directory.

    Args:
        images: dict {frame_idx: BGR image}
        output_dir: path to save annotated images
    """
    import os
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, "detections")

    colors = {
        "power_socket": (0, 0, 255),     # Red
        "ethernet_socket": (255, 0, 0),  # Blue
        "vga_socket": (0, 255, 0),       # Green
    }

    for idx, img in images.items():
        vis = img.copy()
        has_annotation = False

        for entity_name, ann_dict in ENTITY_ANNOTATIONS.items():
            if idx in ann_dict:
                bbox = ann_dict[idx]
                color = colors.get(entity_name, (0, 255, 255))
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
                cv2.putText(vis, entity_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                has_annotation = True

        if has_annotation:
            out_path = os.path.join(output_dir, f"annotated_{idx:06d}.png")
            cv2.imwrite(out_path, vis)
            print(f"  Saved annotated image: {out_path}")


def interactive_annotate(image, frame_idx, entity_name="new_entity"):
    """
    Interactive annotation tool. Opens a window where you can click to
    define bounding box corners.

    Usage on finals day:
        python -c "
        from src.data_loader import load_images
        from src.semantic import interactive_annotate
        imgs = load_images([471])
        interactive_annotate(imgs[471], 471, 'new_entity')
        "

    Returns:
        bbox: [x1, y1, x2, y2] or None if cancelled
    """
    clicks = []
    display = image.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            if len(clicks) == 2:
                x1, y1 = clicks[0]
                x2, y2 = clicks[1]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    window_name = f"Annotate {entity_name} on frame {frame_idx}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, on_click)

    print(f"Click two corners of the bounding box for '{entity_name}'")
    print("Press 'q' when done, 'r' to reset, ESC to cancel")

    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') and len(clicks) >= 2:
            break
        elif key == ord('r'):
            clicks.clear()
            display = image.copy()
        elif key == 27:  # ESC
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)

    x1 = min(clicks[0][0], clicks[1][0])
    y1 = min(clicks[0][1], clicks[1][1])
    x2 = max(clicks[0][0], clicks[1][0])
    y2 = max(clicks[0][1], clicks[1][1])

    bbox = [x1, y1, x2, y2]
    print(f"  Annotation for {entity_name} on frame {frame_idx}: {bbox}")
    return bbox


if __name__ == "__main__":
    import sys
    if "--annotate" in sys.argv:
        from .data_loader import load_images
        images = load_images([471, 496, 515])
        visualize_annotations(images)
        print("Saved annotated images to output/detections/")
