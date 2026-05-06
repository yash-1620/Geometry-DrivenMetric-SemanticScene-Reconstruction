"""
reconstruction.py  —  CP260-2026 Final Project
Sparse Structure-from-Motion reconstruction using SIFT + BFMatcher + DLT.

Pipeline
--------
1. Extract SIFT features from all loaded images.
2. Match features across all image pairs (Lowe ratio test).
3. Triangulate matched keypoints using the provided camera poses.
4. Filter by positive depth + reprojection error threshold.
5. Save result as ASCII PLY point cloud.
"""

import cv2
import numpy as np
from itertools import combinations
from tqdm import tqdm

from .data_loader import get_projection_matrix
from . import config


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(images):
    """
    Extract SIFT keypoints and descriptors from every loaded image.

    Args:
        images   : dict  {frame_idx: BGR numpy array}

    Returns:
        features : dict  {frame_idx: (keypoints, descriptors)}
    """
    sift     = cv2.SIFT_create(nfeatures=config.SIFT_N_FEATURES)
    features = {}

    for idx, img in tqdm(images.items(), desc="Extracting SIFT"):
        gray          = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des       = sift.detectAndCompute(gray, None)
        features[idx] = (kp, des)

    print(f"[SfM] Features extracted from {len(features)} images")
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Feature matching
# ─────────────────────────────────────────────────────────────────────────────

def match_features(features):
    """
    Match SIFT descriptors across every image pair using brute-force L2
    distance and Lowe's ratio test.

    Args:
        features : dict  {frame_idx: (keypoints, descriptors)}

    Returns:
        matches  : dict  {(frame_i, frame_j): list[cv2.DMatch]}
                   Only pairs with >= 30 good matches are kept.
    """
    bf      = cv2.BFMatcher(cv2.NORM_L2)
    matches = {}
    keys    = sorted(features.keys())

    for i, j in combinations(keys, 2):
        _, des1 = features[i]
        _, des2 = features[j]

        if des1 is None or des2 is None:
            continue

        raw  = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in raw
                if m.distance < config.MATCH_RATIO_THRESH * n.distance]

        if len(good) >= 30:
            matches[(i, j)] = good

    print(f"[SfM] Matched pairs: {len(matches)}")
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Triangulation
# ─────────────────────────────────────────────────────────────────────────────

def triangulate_matches(features, matches, poses, K):
    """
    Triangulate all matched keypoint pairs to obtain world-space 3-D points.

    A point is accepted only when:
      * it lies in front of both cameras  (positive depth)
      * reprojection error < config.TRIANGULATION_REPROJ_THRESH in both views

    Args:
        features : dict  {frame_idx: (keypoints, descriptors)}
        matches  : dict  {(frame_i, frame_j): list[cv2.DMatch]}
        poses    : dict  {frame_idx: (4,4) camera-to-world matrix}
        K        : (3,3) camera intrinsic matrix

    Returns:
        points_3d : (N, 3) float64 array  — accepted world-space points
        None      : colour placeholder kept for API compatibility
    """
    points        = []
    reproj_thresh = config.TRIANGULATION_REPROJ_THRESH

    for (i, j), match_list in tqdm(matches.items(), desc="Triangulating"):
        if i not in poses or j not in poses:
            continue

        kp1, _ = features[i]
        kp2, _ = features[j]

        P1 = get_projection_matrix(K, poses[i])
        P2 = get_projection_matrix(K, poses[j])

        pts1 = np.array([kp1[m.queryIdx].pt for m in match_list],
                        dtype=np.float64)
        pts2 = np.array([kp2[m.trainIdx].pt for m in match_list],
                        dtype=np.float64)

        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        for k, pt in enumerate(pts3d):
            if not np.isfinite(pt).all():
                continue

            pt_h  = np.append(pt, 1.0)
            proj1 = P1 @ pt_h
            proj2 = P2 @ pt_h

            # Reject points behind either camera
            if proj1[2] <= 0 or proj2[2] <= 0:
                continue

            err1 = np.linalg.norm(proj1[:2] / proj1[2] - pts1[k])
            err2 = np.linalg.norm(proj2[:2] / proj2[2] - pts2[k])

            if err1 < reproj_thresh and err2 < reproj_thresh:
                points.append(pt)

    if not points:
        print("[SfM] WARNING: no 3D points survived filtering")
        return np.zeros((0, 3)), None

    return np.array(points), None


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_sparse_reconstruction(images, poses, K):
    """
    Run the complete sparse SfM pipeline on a set of posed images.

    Args:
        images : dict  {frame_idx: BGR image}
        poses  : dict  {frame_idx: (4,4) camera-to-world matrix}
        K      : (3,3) intrinsic matrix

    Returns:
        pts3d    : (N, 3) world-space point cloud
        colors   : None  (not computed in this lightweight version)
        features : raw feature dict (reusable downstream)
        matches  : raw match dict
    """
    print("\n=== SfM Reconstruction ===")

    features      = extract_features(images)
    matches       = match_features(features)
    pts3d, colors = triangulate_matches(features, matches, poses, K)

    print(f"[SfM] Total 3D points: {len(pts3d)}")
    return pts3d, colors, features, matches


# ─────────────────────────────────────────────────────────────────────────────
# PLY export
# ─────────────────────────────────────────────────────────────────────────────

def save_point_cloud(points, colors, path):
    """
    Write a point cloud to an ASCII PLY file.

    Args:
        points : (N, 3) XYZ coordinates
        colors : ignored — kept for API compatibility
        path   : destination file path (str)
    """
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

    print(f"[SfM] Saved: {path}")
