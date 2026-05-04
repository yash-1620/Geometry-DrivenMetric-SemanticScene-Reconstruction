"""
Visualization and I/O utilities.
"""
import json
import cv2
import numpy as np


def project_obb_to_image(obb, K, pose_c2w):
    """
    Project an OBB (center, extent, rotation) onto an image.
    Returns the 8 projected corner points (for drawing the bounding box).

    Args:
        obb: dict with 'center' [3], 'extent' [3], 'rotation' [3x3]
        K: 3x3 intrinsic matrix
        pose_c2w: 4x4 camera-to-world pose

    Returns:
        corners_2d: (8, 2) array of projected 2D points
    """
    center = np.array(obb['center'])
    extent = np.array(obb['extent'])
    R_obb = np.array(obb['rotation'])

    # 8 corners of the box in local frame
    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
    ], dtype=np.float64)

    corners_local = signs * extent  # (8, 3)
    # Transform to world frame
    corners_world = (R_obb @ corners_local.T).T + center  # (8, 3)

    # Project to image
    pose_w2c = np.linalg.inv(pose_c2w)
    R_cam = pose_w2c[:3, :3]
    t_cam = pose_w2c[:3, 3]

    corners_cam = (R_cam @ corners_world.T).T + t_cam  # (8, 3)

    # Filter points behind camera
    valid = corners_cam[:, 2] > 0
    if not np.any(valid):
        return None

    corners_2d = np.zeros((8, 2))
    for i in range(8):
        if corners_cam[i, 2] > 0:
            pt = K @ corners_cam[i]
            corners_2d[i] = pt[:2] / pt[2]
        else:
            corners_2d[i] = [np.nan, np.nan]

    return corners_2d


def draw_obb_on_image(image, corners_2d, label="", color=(0, 255, 0), thickness=2):
    """
    Draw a projected OBB wireframe on an image.

    Args:
        image: BGR image (will be modified in place)
        corners_2d: (8, 2) projected corner points
        label: text label to draw
        color: BGR color
        thickness: line thickness
    """
    if corners_2d is None:
        return image

    vis = image.copy()
    pts = corners_2d.astype(int)

    # Draw the 12 edges of the box
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    for i, j in edges:
        if not (np.isnan(pts[i]).any() or np.isnan(pts[j]).any()):
            p1 = tuple(pts[i])
            p2 = tuple(pts[j])
            cv2.line(vis, p1, p2, color, thickness)

    # Draw label
    if label:
        valid_pts = pts[~np.isnan(pts).any(axis=1)]
        if len(valid_pts) > 0:
            top_left = valid_pts.min(axis=0)
            cv2.putText(vis, label, tuple(top_left - [0, 10]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return vis


def draw_bbox_on_image(image, bbox, label="", color=(0, 255, 0), thickness=2):
    """Draw a 2D bounding box [x1, y1, x2, y2] on an image."""
    vis = image.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(vis, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def save_answers_json(entities, output_path):
    """
    Save OBB answers in the exact format required by sample_answers.json.

    Args:
        entities: list of dicts, each with:
            - 'entity': str (e.g. 'power_socket')
            - 'obb': dict with 'center' [3], 'extent' [3], 'rotation' [3x3]
        output_path: path to write JSON
    """
    output = []
    for ent in entities:
        entry = {
            "entity": ent["entity"],
            "obb": {
                "center": [float(v) for v in ent["obb"]["center"]],
                "extent": [float(v) for v in ent["obb"]["extent"]],
                "rotation": [
                    [float(v) for v in row]
                    for row in ent["obb"]["rotation"]
                ]
            }
        }
        output.append(entry)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved answers to {output_path}")


def validate_against_sample(answers, sample_answers):
    """
    Validate pipeline output by comparing VGA socket OBB with sample answer.
    Prints center distance, extent comparison, and rotation alignment.
    """
    # Find VGA socket in both
    sample_vga = None
    our_vga = None
    for s in sample_answers:
        if s['entity'] == 'vga_socket':
            sample_vga = s['obb']
    for a in answers:
        if a['entity'] == 'vga_socket':
            our_vga = a['obb']

    if sample_vga is None:
        print("  [INFO] No VGA socket in sample answers to validate against")
        return
    if our_vga is None:
        print("  [INFO] No VGA socket in our answers to validate")
        return

    # ── Center comparison ───────────────────────────────────────────────
    c_sample = np.array(sample_vga['center'])
    c_ours = np.array(our_vga['center'])
    dist = np.linalg.norm(c_sample - c_ours)
    print(f"\n  === VGA Socket Validation ===")
    print(f"  Center distance: {dist:.4f} m ({dist*1000:.1f} mm)")
    print(f"    Sample:  [{c_sample[0]:.6f}, {c_sample[1]:.6f}, {c_sample[2]:.6f}]")
    print(f"    Ours:    [{c_ours[0]:.6f}, {c_ours[1]:.6f}, {c_ours[2]:.6f}]")

    # ── Extent comparison ───────────────────────────────────────────────
    e_sample = np.array(sample_vga['extent'])
    e_ours = np.array(our_vga['extent'])
    print(f"\n  Extents (half-sizes in meters):")
    print(f"    Sample:  [{e_sample[0]:.6f}, {e_sample[1]:.6f}, {e_sample[2]:.6f}]")
    print(f"    Ours:    [{e_ours[0]:.6f}, {e_ours[1]:.6f}, {e_ours[2]:.6f}]")

    # Compare sorted extents (in case axis ordering differs)
    e_s_sorted = np.sort(e_sample)[::-1]
    e_o_sorted = np.sort(e_ours)[::-1]
    print(f"    Sorted sample: [{e_s_sorted[0]:.6f}, {e_s_sorted[1]:.6f}, {e_s_sorted[2]:.6f}]")
    print(f"    Sorted ours:   [{e_o_sorted[0]:.6f}, {e_o_sorted[1]:.6f}, {e_o_sorted[2]:.6f}]")

    # ── Rotation comparison ─────────────────────────────────────────────
    R_sample = np.array(sample_vga['rotation'])
    R_ours = np.array(our_vga['rotation'])

    # Compute rotation angle between the two
    try:
        R_diff = R_ours @ R_sample.T
        trace = np.clip(np.trace(R_diff), -1.0, 3.0)
        angle_deg = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
        print(f"\n  Rotation angular error: {angle_deg:.1f}°")
    except Exception as e:
        print(f"\n  Rotation comparison error: {e}")

    # Compare individual axes (dot products)
    print(f"  Axis alignment (|dot product|):")
    for i in range(min(3, len(R_sample))):
        for j in range(min(3, len(R_ours))):
            dot = abs(np.dot(R_sample[i], R_ours[j]))
            if dot > 0.8:
                print(f"    Sample axis {i} ~ Our axis {j}: {dot:.4f}")

    # ── Volume comparison ───────────────────────────────────────────────
    vol_sample = np.prod(e_sample) * 8  # full box volume
    vol_ours = np.prod(e_ours) * 8
    print(f"\n  Box volume: sample={vol_sample:.2e} m³, ours={vol_ours:.2e} m³")
    if vol_sample > 0:
        print(f"  Volume ratio: {vol_ours/vol_sample:.2f}x")
