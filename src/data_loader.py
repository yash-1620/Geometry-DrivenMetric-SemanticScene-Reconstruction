"""
Data loading utilities.
Loads images, camera poses (4x4 extrinsics), and intrinsic parameters.
"""
import os
import json
import cv2
import numpy as np
from . import config


def load_intrinsics(path=None):
    """Load camera intrinsic matrix from JSON file."""
    if path is None:
        path = config.INTRINSIC_PATH
    with open(path, 'r') as f:
        data = json.load(f)
    K = np.array(data['camera_matrix'], dtype=np.float64)
    return K


def load_poses(path=None):
    """
    Load all camera poses from poses.json.
    Returns dict: {frame_index(int): 4x4 numpy array (camera-to-world)}.
    """
    if path is None:
        path = config.POSES_PATH
    with open(path, 'r') as f:
        raw = json.load(f)

    poses = {}
    for key, mat in raw.items():
        poses[int(key)] = np.array(mat, dtype=np.float64)
    return poses


def load_images(frame_indices=None, scale=1.0):
    """
    Load images for the specified frame indices.

    Args:
        frame_indices: List of integer frame indices. Defaults to config.FRAME_INDICES.
        scale: Resize factor (1.0 = original size).

    Returns:
        dict: {frame_index: BGR image as numpy array}
    """
    if frame_indices is None:
        frame_indices = config.FRAME_INDICES

    images = {}
    for idx in frame_indices:
        fname = f"frame_{idx:06d}.png"
        fpath = os.path.join(config.DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [WARN] Image not found: {fpath}")
            continue
        img = cv2.imread(fpath)
        if img is None:
            print(f"  [WARN] Failed to read: {fpath}")
            continue
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        images[idx] = img
    return images


def get_scaled_intrinsics(K, scale):
    """Scale the camera intrinsic matrix by a factor."""
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale  # fx
    K_scaled[1, 1] *= scale  # fy
    K_scaled[0, 2] *= scale  # cx
    K_scaled[1, 2] *= scale  # cy
    return K_scaled


def get_projection_matrix(K, pose_c2w):
    """
    Build a 3x4 projection matrix P = K @ [R|t] (world-to-image).

    Args:
        K: 3x3 intrinsic matrix
        pose_c2w: 4x4 camera-to-world transformation

    Returns:
        P: 3x4 projection matrix
    """
    # World-to-camera = inverse of camera-to-world
    pose_w2c = np.linalg.inv(pose_c2w)
    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3:]
    Rt = np.hstack([R, t])
    P = K @ Rt
    return P


def load_sample_answers(path=None):
    """
    Load the sample answers JSON for validation.
    Handles the case where some entries have placeholder values (X, Y, Z, etc.)
    by extracting only the valid VGA socket entry.
    """
    import re
    if path is None:
        path = config.SAMPLE_ANSWERS_PATH

    with open(path, 'r') as f:
        raw = f.read()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Extract the VGA socket entry (which has real values)
    # by finding the first complete object with numeric values
    data = []
    # Replace placeholder letters with 0.0 so the JSON parses
    cleaned = raw
    for placeholder in ['X', 'Y', 'Z', 'W', 'H', 'L', 'rx', 'ry', 'rz']:
        # Replace standalone placeholders (not inside strings)
        cleaned = re.sub(r'(?<!["\w])' + placeholder + r'(?!["\w])', '0.0', cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"  [WARN] Could not parse sample_answers.json: {e}")
        data = []

    return data


def load_dataset():
    """
    Convenience function: load everything at once.

    Returns:
        images: dict {frame_idx: image}
        poses: dict {frame_idx: 4x4 matrix}
        K: 3x3 intrinsic matrix
    """
    print("[1/3] Loading intrinsics...")
    K = load_intrinsics()

    print("[2/3] Loading poses...")
    all_poses = load_poses()
    # Filter to only frames we have images for
    poses = {idx: all_poses[idx] for idx in config.FRAME_INDICES
             if idx in all_poses}

    print("[3/3] Loading images...")
    images = load_images()

    print(f"  Loaded {len(images)} images, {len(poses)} poses")
    return images, poses, K
