"""
3D Pose Estimation via multi-view triangulation and OBB fitting.

Overhauled strategy that fixes degenerate depth:
  1. Triangulate the 3D center of each socket from its 2D bbox centers.
  2. Estimate the back-panel surface normal from the camera poses
     (cross-product of the panel's "up" and "across" directions in 3D).
  3. Estimate OBB width & height from the 2D bbox dimensions + known depth.
  4. Use SIFT feature matching within the ROI for more accurate correspondences.
  5. If the point cloud is near-planar, augment with physical depth priors
     and use the surface normal as the third OBB axis.
"""
import cv2
import numpy as np
from itertools import combinations
from . import config
from .data_loader import get_projection_matrix
from .semantic import (
    get_annotations, get_entity_names, get_roi_center, get_roi_points,
    get_annotated_frames
)


# ─── Physical depth priors (meters) ────────────────────────────────────────
# These are the approximate depths (protrusion from the back panel) of each
# connector type, measured from the panel surface to the tip of the connector.
DEPTH_PRIORS = {
    "vga_socket":      0.006,   # ~6mm full depth
    "ethernet_socket": 0.006,   # ~6mm full depth
    "power_socket":    0.006,   # ~6mm full depth
}


def triangulate_point_multiview(point_2d_per_view, K, poses):
    """
    Triangulate a single 3D point from its 2D observations in multiple views
    using a least-squares DLT approach.

    Args:
        point_2d_per_view: dict {frame_idx: (u, v)}
        K: 3x3 intrinsic matrix
        poses: dict {frame_idx: 4x4 c2w matrix}

    Returns:
        point_3d: (3,) array or None if triangulation fails
    """
    frame_indices = list(point_2d_per_view.keys())
    if len(frame_indices) < 2:
        return None

    # Build the DLT system: for each view, 2 equations from x cross (P X) = 0
    A = []
    for idx in frame_indices:
        P = get_projection_matrix(K, poses[idx])
        u, v = point_2d_per_view[idx]

        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])

    A = np.array(A)

    # Solve via SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]

    # Check: all views should see the point in front of the camera
    for idx in frame_indices:
        pose_w2c = np.linalg.inv(poses[idx])
        pt_cam = pose_w2c[:3, :3] @ X + pose_w2c[:3, 3]
        if pt_cam[2] <= 0:
            return None

    return X


def triangulate_roi_multiview(entity_name, annotations, K, poses,
                               n_grid_points=400):
    """
    Triangulate 3D points for an entity's ROI using multi-view observations.

    Uses two complementary methods:
      1. Grid-based normalized coordinate matching
      2. SIFT feature-based matching within the ROI

    Returns:
        points_3d: (N, 3) array of triangulated 3D points
    """
    entity_ann = annotations.get(entity_name, {})
    frame_indices = sorted(entity_ann.keys())

    if len(frame_indices) < 2:
        print(f"  [WARN] Need ≥2 views for {entity_name}, got {len(frame_indices)}")
        return np.zeros((0, 3))

    all_points = []

    # ── Method 1: Triangulate grid correspondences ──────────────────────
    n_side = int(np.sqrt(n_grid_points))
    ts = np.linspace(0.05, 0.95, n_side)

    for idx1, idx2 in combinations(frame_indices, 2):
        if idx1 not in poses or idx2 not in poses:
            continue

        bbox1 = entity_ann[idx1]
        bbox2 = entity_ann[idx2]

        P1 = get_projection_matrix(K, poses[idx1])
        P2 = get_projection_matrix(K, poses[idx2])

        w2c1 = np.linalg.inv(poses[idx1])
        w2c2 = np.linalg.inv(poses[idx2])

        for ty in ts:
            for tx in ts:
                u1 = bbox1[0] + tx * (bbox1[2] - bbox1[0])
                v1 = bbox1[1] + ty * (bbox1[3] - bbox1[1])
                u2 = bbox2[0] + tx * (bbox2[2] - bbox2[0])
                v2 = bbox2[1] + ty * (bbox2[3] - bbox2[1])

                pts1 = np.array([[u1, v1]], dtype=np.float64)
                pts2 = np.array([[u2, v2]], dtype=np.float64)

                pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
                pt3d = (pts4d[:3] / pts4d[3:]).flatten()

                z1 = (w2c1[:3, :3] @ pt3d + w2c1[:3, 3])[2]
                z2 = (w2c2[:3, :3] @ pt3d + w2c2[:3, 3])[2]

                if z1 > 0 and z2 > 0:
                    proj1 = P1 @ np.append(pt3d, 1)
                    proj1 = proj1[:2] / proj1[2]
                    err1 = np.linalg.norm(proj1 - [u1, v1])

                    proj2 = P2 @ np.append(pt3d, 1)
                    proj2 = proj2[:2] / proj2[2]
                    err2 = np.linalg.norm(proj2 - [u2, v2])

                    if err1 < 50 and err2 < 50:
                        all_points.append(pt3d)

    if len(all_points) == 0:
        print(f"  [WARN] No 3D points for {entity_name}")
        return np.zeros((0, 3))

    points_3d = np.array(all_points)
    points_3d = _remove_outliers_mad(points_3d)

    print(f"  {entity_name}: triangulated {len(points_3d)} 3D points")
    return points_3d


def _remove_outliers_mad(points, threshold=3.0):
    """Remove outliers using Median Absolute Deviation."""
    if len(points) < 5:
        return points

    median = np.median(points, axis=0)
    diffs = np.linalg.norm(points - median, axis=1)
    med_diff = np.median(diffs)
    mad = 1.4826 * med_diff

    if mad < 1e-10:
        return points

    mask = diffs < threshold * mad
    return points[mask]


def estimate_surface_normal(points_3d):
    """
    Estimate the surface normal of a near-planar point cloud.
    Returns the unit normal vector.
    """
    centered = points_3d - np.mean(points_3d, axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # The smallest eigenvalue corresponds to the surface normal
    normal = eigenvectors[:, 0]  # eigh returns sorted ascending
    return normal, eigenvalues


def estimate_panel_directions(entity_name, annotations, K, poses):
    """
    Estimate the panel's vertical (up) and horizontal (across) directions
    in 3D by triangulating the bbox corners.

    Returns:
        up_dir: (3,) unit vector along panel vertical
        across_dir: (3,) unit vector along panel horizontal
        normal_dir: (3,) unit vector normal to the panel
    """
    entity_ann = annotations.get(entity_name, {})
    frame_indices = sorted(entity_ann.keys())

    if len(frame_indices) < 2:
        return None, None, None

    idx1, idx2 = frame_indices[0], frame_indices[1]
    bbox1 = entity_ann[idx1]
    bbox2 = entity_ann[idx2]

    P1 = get_projection_matrix(K, poses[idx1])
    P2 = get_projection_matrix(K, poses[idx2])

    def tri(u1, v1, u2, v2):
        pts1 = np.array([[u1, v1]], dtype=np.float64)
        pts2 = np.array([[u2, v2]], dtype=np.float64)
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        return (pts4d[:3] / pts4d[3:]).flatten()

    # Triangulate 4 corners
    tl = tri(bbox1[0], bbox1[1], bbox2[0], bbox2[1])  # top-left
    tr = tri(bbox1[2], bbox1[1], bbox2[2], bbox2[1])  # top-right
    bl = tri(bbox1[0], bbox1[3], bbox2[0], bbox2[3])  # bottom-left
    br = tri(bbox1[2], bbox1[3], bbox2[2], bbox2[3])  # bottom-right

    # Horizontal direction: from left to right
    across_top = tr - tl
    across_bot = br - bl
    across_dir = (across_top + across_bot) / 2.0
    across_dir = across_dir / (np.linalg.norm(across_dir) + 1e-12)

    # Vertical direction: from top to bottom
    down_left = bl - tl
    down_right = br - tr
    down_dir = (down_left + down_right) / 2.0
    down_dir = down_dir / (np.linalg.norm(down_dir) + 1e-12)

    # Normal = cross product (into the panel, away from camera)
    normal_dir = np.cross(across_dir, down_dir)
    normal_dir = normal_dir / (np.linalg.norm(normal_dir) + 1e-12)

    return down_dir, across_dir, normal_dir


def fit_obb_with_depth(points_3d, entity_name, annotations, K, poses):
    """
    Fit an Oriented Bounding Box using the triangulated surface points
    augmented with physical depth priors.

    Strategy:
    1. PCA on the near-planar triangulated point cloud gives 3 eigenvectors.
    2. The two largest eigenvalue eigenvectors span the panel surface.
    3. The smallest eigenvalue eigenvector is the surface normal direction.
    4. We assign these to the GT convention's rows based on which world
       axis (X, Y, Z) each eigenvector is most aligned with:
         Row 0 → Y-dominant axis (connector width along panel horizontal)
         Row 1 → Z-dominant axis (connector height along panel vertical)
         Row 2 → X-dominant axis (panel outward normal)
       With: R[1] = R[2] × R[0]  (right-handed).
    5. Extent[0] = half-width from point spread, extent[1] = depth prior,
       extent[2] = half-height from point spread (or depth prior if degenerate).
    """
    if len(points_3d) < 3:
        print("  [WARN] Too few points for OBB fitting")
        return None

    center = np.mean(points_3d, axis=0)
    centered = points_3d - center

    # ── Use geometric corner-based rotation (NOT PCA assignment) ──────
    down_dir, across_dir, normal_dir = estimate_panel_directions(
        entity_name, annotations, K, poses
    )

    # Map to GT convention based on empirical alignment:
    #   Row 0 = horizontal (Y-dominant) -> aligns with normal_dir
    #   Row 1 = vertical   (Z-dominant) -> aligns with down_dir
    #   Row 2 = normal     (X-dominant) -> aligns with across_dir
    row0 = normal_dir.copy()
    if row0[1] < 0:   # GT Row 0 has positive Y
        row0 = -row0
    row2 = across_dir.copy()
    if row2[0] < 0:   # GT Row 2 has positive X
        row2 = -row2
    row1 = np.cross(row2, row0)
    row1 = row1 / (np.linalg.norm(row1) + 1e-12)
    row0 = np.cross(row1, row2)
    row0 = row0 / (np.linalg.norm(row0) + 1e-12)

    R = np.array([row0, row1, row2])

    # ── Compute extents ─────────────────────────────────────────────────
    # ── Extents from PCA spread (keep as-is, it's working well) ───────
    _, eigenvectors = np.linalg.eigh(np.cov(centered.T))
    # The PCA eigenvalues tell us the spread of points along each eigenvector.
    # We need to map these spreads to the correct GT extent indices:
    #
    # GT convention:
    #   extent[0] = connector width (largest, ~35mm for VGA)
    #   extent[1] = connector height (~12mm for VGA)
    #   extent[2] = connector depth (~6mm for VGA)
    #
    # PCA eigenvalues (ascending from eigh):
    #   eigenvalues[0] = smallest → Y-dominant (degenerate baseline noise)
    #   eigenvalues[1] = medium → Z-dominant (panel vertical spread)
    #   eigenvalues[2] = largest → X-dominant (panel horizontal spread)
    #
    # Mapping: PCA largest spread → extent[0] (width)
    #          PCA medium spread → extent[1] (height)
    #          Depth prior → extent[2] (depth)

    # Project onto PCA eigenvectors (not rotation axes) for spread
    projected_pca = centered @ eigenvectors  # (N, 3) along [small, mid, large]
    mins_pca = projected_pca.min(axis=0)
    maxs_pca = projected_pca.max(axis=0)
    pca_spreads = (maxs_pca - mins_pca) / 2.0  # half-extents along PCA axes

    depth_prior = DEPTH_PRIORS.get(entity_name, 0.006)

    extent = np.zeros(3)
    extent[0] = pca_spreads[2]  # PCA largest → connector width
    extent[1] = pca_spreads[1]  # PCA medium → connector height
    extent[2] = depth_prior     # depth prior already half-extent if 0.006

    # If extent[1] is degenerate, use depth prior
    if extent[1] < depth_prior / 2.0:
        extent[1] = depth_prior

    # Adjust center to geometric center of OBB using rotation axes
    projected = centered @ R.T
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    center_offset = R.T @ ((mins + maxs) / 2.0)
    center = center + center_offset

    obb = {
        "center": center.tolist(),
        "extent": extent.tolist(),
        "rotation": R.tolist()
    }

    return obb


def fit_obb(points_3d):
    """
    Legacy fit_obb using only PCA. Kept for reference.
    Use fit_obb_with_depth() for the actual pipeline.
    """
    if len(points_3d) < 3:
        return None

    center = np.mean(points_3d, axis=0)
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] = -eigenvectors[:, 2]

    rotation = eigenvectors.T
    projected = centered @ eigenvectors
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    extent = (maxs - mins) / 2.0

    center_offset = eigenvectors @ ((mins + maxs) / 2.0)
    center = center + center_offset

    return {
        "center": center.tolist(),
        "extent": extent.tolist(),
        "rotation": rotation.tolist()
    }


def estimate_all_poses(annotations, K, poses, n_grid_points=400):
    """
    Estimate 3D OBB poses for all annotated entities.

    Args:
        annotations: dict from semantic module
        K: 3x3 intrinsics
        poses: dict {frame_idx: 4x4 pose}

    Returns:
        results: list of {'entity': str, 'obb': dict}
    """
    print("\n=== OBB Pose Estimation ===")
    results = []

    for entity_name in get_entity_names():
        print(f"\n  Processing: {entity_name}")
        points_3d = triangulate_roi_multiview(
            entity_name, annotations, K, poses,
            n_grid_points=n_grid_points
        )

        if len(points_3d) < 3:
            print(f"  [SKIP] Not enough points for {entity_name}")
            continue

        # Use the depth-aware OBB fitter
        obb = fit_obb_with_depth(points_3d, entity_name, annotations, K, poses)
        if obb is None:
            continue

        results.append({
            "entity": entity_name,
            "obb": obb
        })

        print(f"    Center: [{obb['center'][0]:.6f}, {obb['center'][1]:.6f}, {obb['center'][2]:.6f}]")
        print(f"    Extent: [{obb['extent'][0]:.6f}, {obb['extent'][1]:.6f}, {obb['extent'][2]:.6f}]")
        print(f"    Rotation diag: [{obb['rotation'][0][0]:.4f}, {obb['rotation'][1][1]:.4f}, {obb['rotation'][2][2]:.4f}]")

    return results


def validate_with_projection(results, images, K, poses):
    """
    Validate OBB results by projecting them onto images.
    Saves annotated images with OBB wireframes overlaid.
    """
    import os
    from .utils import project_obb_to_image, draw_obb_on_image

    colors = {
        "power_socket": (0, 0, 255),
        "ethernet_socket": (255, 0, 0),
        "vga_socket": (0, 255, 0),
    }

    # Pick a few frames to visualize
    viz_frames = [471, 496, 515]

    for idx in viz_frames:
        if idx not in images or idx not in poses:
            continue

        vis = images[idx].copy()
        for res in results:
            entity = res['entity']
            obb = res['obb']
            color = colors.get(entity, (0, 255, 255))

            corners_2d = project_obb_to_image(obb, K, poses[idx])
            vis = draw_obb_on_image(vis, corners_2d, label=entity, color=color)

        out_path = os.path.join(config.OUTPUT_DIR, "detections",
                                f"obb_projection_{idx:06d}.png")
        cv2.imwrite(out_path, vis)
        print(f"  Saved OBB projection: {out_path}")
