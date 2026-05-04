#!/usr/bin/env python3
"""
Metric-Semantic Reconstruction Pipeline — CP260-2026 Final Project

Master script that runs the full pipeline:
  1. Load data (images, poses, intrinsics)
  2. Build sparse 3D reconstruction
  3. Annotate semantic entities (power/ethernet sockets)
  4. Estimate 3D OBB poses via multi-view triangulation
  5. Validate by projecting OBBs onto images
  6. Save answers.json in submission format

Usage:
    conda run -n ml python run_pipeline.py
    conda run -n ml python run_pipeline.py --skip-reconstruction
    conda run -n ml python run_pipeline.py --validate-only
"""
import os
import sys
import json
import time
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.data_loader import load_dataset, load_sample_answers
from src.reconstruction import build_sparse_reconstruction, save_point_cloud
from src.semantic import get_annotations, visualize_annotations
from src.pose_estimation import estimate_all_poses, validate_with_projection
from src.utils import save_answers_json, validate_against_sample


def main():
    parser = argparse.ArgumentParser(
        description="Metric-Semantic 3D Reconstruction Pipeline"
    )
    parser.add_argument("--skip-reconstruction", action="store_true",
                        help="Skip sparse reconstruction, go straight to pose estimation")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validation on existing answers.json")
    parser.add_argument("--grid-points", type=int, default=400,
                        help="Number of grid points per ROI for triangulation")
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 60)
    print("  Metric-Semantic 3D Reconstruction Pipeline")
    print("  CP260-2026 Final Project")
    print("=" * 60)

    # ── Step 1: Load Data ─────────────────────────────────────────────────
    print("\n[Step 1] Loading dataset...")
    images, poses, K = load_dataset()

    # ── Step 2: Sparse Reconstruction (optional) ─────────────────────────
    if not args.skip_reconstruction and not args.validate_only:
        print("\n[Step 2] Building sparse 3D reconstruction...")
        points_3d, colors, features, matches = \
            build_sparse_reconstruction(images, poses, K)

        if len(points_3d) > 0:
            ply_path = os.path.join(config.OUTPUT_DIR, "point_cloud.ply")
            save_point_cloud(points_3d, colors, ply_path)
    else:
        print("\n[Step 2] Skipping reconstruction")

    # ── Step 3: Semantic Annotations ─────────────────────────────────────
    print("\n[Step 3] Loading semantic annotations...")
    annotations = get_annotations()
    print(f"  Entities: {list(annotations.keys())}")
    for name, ann in annotations.items():
        print(f"    {name}: annotated in frames {list(ann.keys())}")

    # Save annotated images for visualization
    visualize_annotations(images)

    # ── Step 4: OBB Pose Estimation ──────────────────────────────────────
    if not args.validate_only:
        print("\n[Step 4] Estimating 3D poses...")
        results = estimate_all_poses(
            annotations, K, poses,
            n_grid_points=args.grid_points
        )

        # ── Step 5: Save Results ─────────────────────────────────────────
        print("\n[Step 5] Saving results...")
        answers_path = os.path.join(config.OUTPUT_DIR, "answers.json")
        save_answers_json(results, answers_path)

        # ── Step 6: Validate ─────────────────────────────────────────────
        print("\n[Step 6] Validating...")

        # Project OBBs onto images
        validate_with_projection(results, images, K, poses)

        # Compare VGA socket with sample answer
        sample_answers = load_sample_answers()
        validate_against_sample(results, sample_answers)

    else:
        # Validate existing answers
        answers_path = os.path.join(config.OUTPUT_DIR, "answers.json")
        if os.path.exists(answers_path):
            with open(answers_path, 'r') as f:
                results = json.load(f)
            sample_answers = load_sample_answers()
            validate_against_sample(results, sample_answers)
            validate_with_projection(results, images, K, poses)
        else:
            print(f"  [ERROR] No answers.json found at {answers_path}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Results saved to: {config.OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
