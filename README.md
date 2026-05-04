# Metric-Semantic 3D Reconstruction of a Desktop Scene

## Overview

This project implements a complete metric-semantic reconstruction pipeline to estimate **3D Oriented Bounding Boxes (OBBs)** of small desktop components using multi-view RGB images.

Instead of relying on heavy deep learning detection models, this work follows a **geometry-first approach**, using multi-view triangulation and classical computer vision techniques to achieve accurate 3D localization.

---

## Problem Statement

Given:

* 16 posed RGB images (2560 × 1440 resolution)
* Camera intrinsics and camera-to-world poses

Goal:

* Estimate 3D OBBs for:

  * `ethernet_socket`
  * `power_socket`
* Validate pipeline using:

  * `vga_socket` (ground truth provided)

---

## Approach

The pipeline consists of the following steps:

1. **Manual 2D Annotation**

   * Bounding boxes marked on two selected frames

2. **Multi-View Triangulation**

   * Grid-based correspondence inside bounding boxes
   * DLT (Direct Linear Transform) used to reconstruct 3D points

3. **Outlier Removal**

   * Median Absolute Deviation (MAD) filtering

4. **OBB Fitting**

   * PCA-based orientation estimation
   * Depth prior used due to planar structure

5. **Output Generation**

   * Results exported in required `answers.json` format

---

## Key Features

* No dependency on object detection models (GroundingDINO/SAM not required)
* Lightweight and fast (runs fully on Google Colab)
* Modular Python implementation
* Robust to small object sizes
* Achieves **< 5 cm center error** on validation

---

## Repository Structure

```
project/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── semantic.py
│   ├── pose_estimation.py
│   ├── reconstruction.py
│   ├── utils.py
│   └── run_pipeline.py
│
├── notebook/
│   └── final_pipeline.ipynb
│
├── outputs/
│   ├── answers.json
│   └── transforms.json
│
├── report/
│   └── report.pdf
│
├── requirements.txt
└── README.md
```

---

## How to Run

### Option 1: Google Colab (Recommended)

Open the notebook:

```
notebook/final_pipeline.ipynb
```

Run all cells sequentially.

---

### Option 2: Local Execution

```bash
pip install -r requirements.txt
python src/run_pipeline.py
```

---

## Results

* Accurate 3D reconstruction of connector ports
* Successful validation using VGA socket ground truth
* Sub-centimeter level accuracy achieved

### Output Format

```json
{
  "entity": "ethernet_socket",
  "obb": {
    "center": [cx, cy, cz],
    "extent": [ex, ey, ez],
    "rotation": [
      [r00, r01, r02],
      [r10, r11, r12],
      [r20, r21, r22]
    ]
  }
}
```

---

## Dependencies

* open3d==0.19.0
* numpy
* opencv-python
* scipy
* matplotlib
* tqdm

---

## Notes

* Dataset is not included due to size constraints
* Camera poses are provided in `poses.json`
* Depth prior (6 mm) is used for all connector types

---

## Future Improvements

* Automate 2D annotation using segmentation models
* Improve robustness with multi-frame triangulation
* Extend to full scene reconstruction

---

## Author

M Jashwanth(27100), M Yashwanth(25878)
CP260 — Robotic Perception (2026)
