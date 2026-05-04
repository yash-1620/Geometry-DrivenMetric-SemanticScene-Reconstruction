"""
Configuration for the Metric-Semantic Reconstruction Pipeline.
CP260-2026 Final Project
"""
import os
import numpy as np

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
INTRINSIC_PATH = os.path.join(PROJECT_ROOT, "intrinsic.json")
POSES_PATH = os.path.join(DATA_DIR, "poses.json")
SAMPLE_ANSWERS_PATH = os.path.join(PROJECT_ROOT, "sample_answers.json")

# ─── Camera Intrinsics (from intrinsic.json) ─────────────────────────────────
IMAGE_WIDTH = 2560
IMAGE_HEIGHT = 1440
FX = 1477.00974684544
FY = 1480.4424455584467
CX = 1298.2501500778505
CY = 686.8201623541711

CAMERA_MATRIX = np.array([
    [FX, 0.0, CX],
    [0.0, FY, CY],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# ─── Frame indices (extracted from filenames in Data/) ───────────────────────
FRAME_INDICES = [319, 333, 353, 359, 365, 371, 390, 400,
                 426, 449, 461, 468, 471, 496, 515, 531]

# ─── Processing parameters ──────────────────────────────────────────────────
SCALE_FACTOR = 0.5  # Downscale for feature matching speed
SIFT_N_FEATURES = 8000
MATCH_RATIO_THRESH = 0.75
TRIANGULATION_REPROJ_THRESH = 4.0  # pixels

# ─── Reconstruction parameters ──────────────────────────────────────────────
VOXEL_SIZE = 0.002          # 2mm voxel downsampling for dense cloud
SOR_NB_NEIGHBORS = 30       # Statistical outlier removal
SOR_STD_RATIO = 1.5

# ─── Ensure output directories exist ────────────────────────────────────────
for subdir in ["", "detections", "depth_maps", "visualizations"]:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)
