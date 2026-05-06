#!/usr/bin/env python3
"""
run_pipeline.py  —  CP260-2026 Final Project
Master orchestration script for the Metric-Semantic 3D Reconstruction Pipeline.

Steps
-----
1.  Load images, camera poses, and intrinsics.
2.  (Optional) Build sparse SfM point cloud for scene context.
3.  Load per-entity 2-D bounding-box annotations derived from known
    world centres (SOCKET_WORLD_CENTERS in semantic.py).
4.  Estimate 3-D Oriented Bounding Boxes via multi-view DLT triangulation.
5.  Save answers.json in the required submission format.
6.  Validate against the VGA socket ground truth.

Usage
-----
    # Full pipeline (SfM + pose estimation)
    python -m src.run_pipeline

    # Skip slow SfM, jump straight to pose estimation
    python -m src.run_pipeline --skip-reconstruction

    # Validate a previously generated answers.json only
    python -m src.run_pipeline --validate-only

    # Tune grid density (default 400; 800 recommended)
    python -m src.run_pipeline --grid-points 800

    # Skip saving annotated detection images (faster iteration)
    python -m src.run_pipeline --no-visualize
"""

import os
import sys
import json
import time
import argparse

# Allow execution as `python run_pipeline.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.data_loader     import load_dataset, load_sample_answers
from src.reconstruction  import build_sparse_reconstruction, save_point_cloud
from src.semantic        import get_annotations, visualize_annotations
from src.pose_estimation import estimate_all_poses, validate_with_projection
from src.utils           import save_answers_json, validate_against_sample


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="CP260-2026 — Metric-Semantic 3D Reconstruction Pipeline"
    )
    p.add_argument(
        "--skip-reconstruction", action="store_true",
        help="Skip sparse SfM (saves ~8 s; uses provided poses directly)."
    )
    p.add_argument(
        "--validate-only", action="store_true",
        help="Skip all estimation; load existing output/answers.json and "
             "run validation only."
    )
    p.add_argument(
        "--grid-points", type=int, default=400, metavar="N",
        help="Approximate grid points sampled per ROI per view pair "
             "(default: 400; use 800 for best accuracy)."
    )
    p.add_argument(
        "--no-visualize", action="store_true",
        help="Skip writing annotated detection images to output/detections/."
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def _step_load(args):
    """Step 1 — Load all data."""
    print("\n[Step 1] Loading dataset...")
    images, poses, K = load_dataset()
    print(f"  Images : {len(images)}  |  Poses : {len(poses)}")
    return images, poses, K


def _step_sfm(images, poses, K, args):
    """Step 2 — Optional sparse SfM reconstruction."""
    if args.skip_reconstruction or args.validate_only:
        print("\n[Step 2] Sparse SfM skipped.")
        return

    print("\n[Step 2] Building sparse 3D reconstruction...")
    pts3d, colors, _, _ = build_sparse_reconstruction(images, poses, K)

    if len(pts3d) > 0:
        ply_path = os.path.join(config.OUTPUT_DIR, "point_cloud.ply")
        save_point_cloud(pts3d, colors, ply_path)
    else:
        print("  [WARN] No 3D points produced — check image/pose alignment.")


def _step_annotate(images, args):
    """Step 3 — Load semantic annotations and optionally visualise them."""
    print("\n[Step 3] Loading semantic annotations...")
    annotations = get_annotations()

    print(f"  Entities : {list(annotations.keys())}")
    for name, ann in annotations.items():
        print(f"    {name}: frames {sorted(ann.keys())}")

    if not args.no_visualize:
        visualize_annotations(images)

    return annotations


def _step_estimate(annotations, K, poses, args):
    """Step 4 — Multi-view triangulation + OBB fitting."""
    print("\n[Step 4] Estimating 3D poses...")
    results = estimate_all_poses(
        annotations, K, poses,
        n_grid_points=args.grid_points
    )
    return results


def _step_save(results):
    """Step 5 — Persist answers.json."""
    print("\n[Step 5] Saving results...")
    answers_path = os.path.join(config.OUTPUT_DIR, "answers.json")
    save_answers_json(results, answers_path)
    return answers_path


def _step_validate(results, images, K, poses):
    """Step 6 — Validate against VGA socket ground truth."""
    print("\n[Step 6] Validating...")
    validate_with_projection(results, images, K, poses)

    sample_answers = load_sample_answers()
    validate_against_sample(results, sample_answers)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args       = _parse_args()
    start_time = time.time()

    _banner("Metric-Semantic 3D Reconstruction Pipeline — CP260-2026")

    # ── Step 1: Load ──────────────────────────────────────────────────────
    images, poses, K = _step_load(args)

    # ── Step 2: SfM (optional) ────────────────────────────────────────────
    _step_sfm(images, poses, K, args)

    # ── Step 3: Annotations ───────────────────────────────────────────────
    annotations = _step_annotate(images, args)

    # ── Steps 4-5: Estimate + save  OR  load existing ────────────────────
    if args.validate_only:
        answers_path = os.path.join(config.OUTPUT_DIR, "answers.json")
        if not os.path.exists(answers_path):
            print(f"\n  [ERROR] No answers.json at {answers_path}\n"
                  f"  Run without --validate-only first.")
            sys.exit(1)
        with open(answers_path, "r") as f:
            results = json.load(f)
        print(f"\n[Step 4-5] Loaded existing answers: {answers_path}")
    else:
        results      = _step_estimate(annotations, K, poses, args)
        answers_path = _step_save(results)

    # ── Step 6: Validate ──────────────────────────────────────────────────
    _step_validate(results, images, K, poses)

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    _banner(f"Pipeline complete in {elapsed:.1f}s  |  Output: {config.OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(msg):
    width = max(60, len(msg) + 4)
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


if __name__ == "__main__":
    main()
