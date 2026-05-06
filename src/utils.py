"""
Visualization and I/O utilities.
"""
import json
import cv2
import numpy as np
import os


def project_obb_to_image(obb, K, pose_c2w):
    """Project OBB 8 corners onto the image plane. Returns an (8,2) array."""
    center  = np.array(obb['center'])
    extent  = np.array(obb['extent'])
    R_obb   = np.array(obb['rotation'])
    
    # 8 corners of a 3D box
    signs   = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [ 1, -1, -1], [ 1, -1, 1], [ 1, 1, -1], [ 1, 1, 1]
    ], dtype=np.float64)
    
    # Calculate corner coordinates in world space
    corners_world = (R_obb @ (signs * extent).T).T + center
    
    # Convert from world to camera space
    pose_w2c = np.linalg.inv(pose_c2w)
    corners_cam = (pose_w2c[:3, :3] @ corners_world.T).T + pose_w2c[:3, 3]
    
    corners_2d = np.full((8, 2), np.nan)
    
    # Project valid (in front of camera) points onto the 2D image plane
    for i in range(8):
        if corners_cam[i, 2] > 0:
            pt = K @ corners_cam[i]
            corners_2d[i] = pt[:2] / pt[2]
            
    return corners_2d


def draw_obb_on_image(image, corners_2d, label="", color=(0, 255, 0), thickness=2):
    """Draw OBB wireframe (12 edges) on a copy of the image."""
    if corners_2d is None:
        return image
        
    vis  = image.copy()
    pts  = corners_2d.astype(int)
    
    # Indices to connect the 8 corners into a box
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), 
        (1, 5), (2, 3), (2, 6), (3, 7), 
        (4, 5), (4, 6), (5, 7), (6, 7)
    ]
    
    for i, j in edges:
        if not (np.isnan(corners_2d[i]).any() or np.isnan(corners_2d[j]).any()):
            cv2.line(vis, tuple(pts[i]), tuple(pts[j]), color, thickness)
            
    if label:
        valid = pts[~np.isnan(corners_2d).any(axis=1)]
        if len(valid) > 0:
            tl = valid.min(axis=0)
            cv2.putText(vis, label, tuple(tl - [0, 10]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
    return vis


def save_answers_json(results, path):
    """Save OBB answers in the required submission format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved answers to {path}")


def validate_against_sample(results, sample_answers):
    """Compare the VGA socket prediction against the ground truth."""
    if not sample_answers:
        print("  [INFO] No sample answers available")
        return

    ours = next((r["obb"] for r in results if r["entity"] == "vga_socket"), None)
    sample = next((r["obb"] for r in sample_answers if r.get("entity") == "vga_socket"), None)

    if ours is None:
        print("  [INFO] No VGA socket in our answers to validate")
        return

    if sample is None:
        print("  [INFO] No VGA socket in sample answers")
        return

    print("\n  === VGA Socket Validation ===")

    ours_c = np.array(ours["center"])
    sample_c = np.array(sample["center"])

    dist = np.linalg.norm(ours_c - sample_c)

    print(f"  Center distance: {dist:.4f} m ({dist*1000:.1f} mm)")
    print(f"    Sample GT: {np.round(sample_c, 4).tolist()}")
    print(f"    Our Pred:  {np.round(ours_c, 4).tolist()}")
    
    e_s = np.sort(np.array(sample["extent"]))[::-1]
    e_o = np.sort(np.array(ours["extent"]))[::-1]
    
    print(f"  Extents (sorted): GT={np.round(e_s, 4).tolist()}  Pred={np.round(e_o, 4).tolist()}")
