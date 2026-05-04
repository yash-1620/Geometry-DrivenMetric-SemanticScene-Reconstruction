"""
3D Reconstruction via multi-view feature matching and triangulation.
Builds a sparse 3D point cloud from SIFT features across multiple posed views.
"""
import cv2
import numpy as np
from itertools import combinations
from tqdm import tqdm
from . import config
from .data_loader import get_projection_matrix, get_scaled_intrinsics


def extract_features(images, n_features=None, scale=None):
    """
    Extract SIFT features from all images.

    Args:
        images: dict {frame_idx: BGR image}
        n_features: max features per image
        scale: resize factor for feature extraction

    Returns:
        features: dict {frame_idx: (keypoints, descriptors)}
    """
    if n_features is None:
        n_features = config.SIFT_N_FEATURES
    if scale is None:
        scale = config.SCALE_FACTOR

    sift = cv2.SIFT_create(nfeatures=n_features)
    features = {}

    for idx, img in tqdm(images.items(), desc="  Extracting SIFT features"):
        if scale != 1.0:
            img_scaled = cv2.resize(img, None, fx=scale, fy=scale)
        else:
            img_scaled = img

        gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
        kps, descs = sift.detectAndCompute(gray, None)

        # Scale keypoints back to original image coordinates
        if scale != 1.0:
            for kp in kps:
                kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)

        features[idx] = (kps, descs)

    return features


def match_features(features, ratio_thresh=None):
    """
    Match SIFT features between all pairs of images.

    Returns:
        matches: dict {(idx1, idx2): list of DMatch}
    """
    if ratio_thresh is None:
        ratio_thresh = config.MATCH_RATIO_THRESH

    bf = cv2.BFMatcher(cv2.NORM_L2)
    frame_indices = sorted(features.keys())
    matches = {}

    pairs = list(combinations(frame_indices, 2))
    for idx1, idx2 in tqdm(pairs, desc="  Matching feature pairs"):
        _, desc1 = features[idx1]
        _, desc2 = features[idx2]

        if desc1 is None or desc2 is None:
            continue
        if len(desc1) < 2 or len(desc2) < 2:
            continue

        raw_matches = bf.knnMatch(desc1, desc2, k=2)

        # Lowe's ratio test
        good = []
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)

        if len(good) >= 10:
            matches[(idx1, idx2)] = good

    print(f"  Found {len(matches)} valid image pairs")
    return matches


def triangulate_matches(features, matches, poses, K):
    """
    Triangulate 3D points from matched features using known camera poses.

    Args:
        features: dict {frame_idx: (keypoints, descriptors)}
        matches: dict {(idx1, idx2): list of DMatch}
        poses: dict {frame_idx: 4x4 c2w matrix}
        K: 3x3 intrinsic matrix

    Returns:
        points_3d: (N, 3) array of 3D world points
        colors: (N, 3) array of BGR colors
    """
    all_points = []
    all_colors = []
    reproj_thresh = config.TRIANGULATION_REPROJ_THRESH

    for (idx1, idx2), match_list in tqdm(matches.items(),
                                          desc="  Triangulating"):
        if idx1 not in poses or idx2 not in poses:
            continue

        kps1, _ = features[idx1]
        kps2, _ = features[idx2]

        P1 = get_projection_matrix(K, poses[idx1])
        P2 = get_projection_matrix(K, poses[idx2])

        pts1 = np.array([kps1[m.queryIdx].pt for m in match_list],
                        dtype=np.float64)
        pts2 = np.array([kps2[m.trainIdx].pt for m in match_list],
                        dtype=np.float64)

        # Triangulate
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)  # (4, N)
        pts3d = (pts4d[:3] / pts4d[3:]).T  # (N, 3)

        # Filter by reprojection error
        for i, pt3d in enumerate(pts3d):
            pt_h = np.append(pt3d, 1.0)

            # Reproject to view 1
            proj1 = P1 @ pt_h
            if proj1[2] <= 0:
                continue
            proj1_2d = proj1[:2] / proj1[2]
            err1 = np.linalg.norm(proj1_2d - pts1[i])

            # Reproject to view 2
            proj2 = P2 @ pt_h
            if proj2[2] <= 0:
                continue
            proj2_2d = proj2[:2] / proj2[2]
            err2 = np.linalg.norm(proj2_2d - pts2[i])

            if err1 < reproj_thresh and err2 < reproj_thresh:
                all_points.append(pt3d)
                # Use color from first image (no need to load here, use white)
                all_colors.append([200, 200, 200])

    if len(all_points) == 0:
        print("  [WARN] No points triangulated!")
        return np.zeros((0, 3)), np.zeros((0, 3))

    points_3d = np.array(all_points)
    colors = np.array(all_colors, dtype=np.uint8)

    print(f"  Triangulated {len(points_3d)} 3D points")
    return points_3d, colors


def filter_point_cloud(points_3d, colors=None, voxel_size=None,
                        sor_neighbors=None, sor_std=None):
    """
    Filter and clean a point cloud using voxel downsampling and
    statistical outlier removal.
    """
    import open3d as o3d

    if voxel_size is None:
        voxel_size = config.VOXEL_SIZE
    if sor_neighbors is None:
        sor_neighbors = config.SOR_NB_NEIGHBORS
    if sor_std is None:
        sor_std = config.SOR_STD_RATIO

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    if colors is not None and len(colors) == len(points_3d):
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(float) / 255.0)

    # Voxel downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Statistical outlier removal
    pcd_clean, _ = pcd_down.remove_statistical_outlier(
        nb_neighbors=sor_neighbors, std_ratio=sor_std
    )

    points_out = np.asarray(pcd_clean.points)
    colors_out = (np.asarray(pcd_clean.colors) * 255).astype(np.uint8) \
        if pcd_clean.has_colors() else None

    print(f"  Filtered: {len(points_3d)} → {len(points_out)} points")
    return points_out, colors_out


def save_point_cloud(points_3d, colors, path):
    """Save point cloud as PLY file using Open3D."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(float) / 255.0)

    o3d.io.write_point_cloud(path, pcd)
    print(f"  Saved point cloud to {path} ({len(points_3d)} points)")


def build_sparse_reconstruction(images, poses, K):
    """
    Full sparse reconstruction pipeline.

    Returns:
        points_3d: (N, 3) filtered 3D points
        colors: (N, 3) point colors
        features: extracted features dict
        matches: feature matches dict
    """
    print("\n=== Sparse 3D Reconstruction ===")

    # Extract features
    features = extract_features(images)

    # Match features
    matches = match_features(features)

    # Triangulate
    points_3d, colors = triangulate_matches(features, matches, poses, K)

    if len(points_3d) == 0:
        return points_3d, colors, features, matches

    # Filter
    points_3d, colors = filter_point_cloud(points_3d, colors)

    return points_3d, colors, features, matches
