# Metric-Semantic 3D Reconstruction Pipeline

**CP260-2026 Final Project**

##  Overview

This project implements a **metric-semantic 3D reconstruction pipeline** that estimates **3D Oriented Bounding Boxes (OBB)** of objects (VGA socket, Ethernet socket, Power socket) from multi-view images.

The pipeline combines:

* Sparse Structure-from-Motion (SfM)
* Semantic annotation (2D bounding boxes)
* Multi-view triangulation
* PCA-based 3D bounding box fitting

---

##  Pipeline Steps

### 1. Dataset Loading

* Loads camera intrinsics (`K`)
* Loads camera poses (`poses.json`)
* Loads RGB images

---

### 2. Sparse 3D Reconstruction (SfM)

* Extracts SIFT features
* Matches features across image pairs
* Triangulates 3D points
* Filters using reprojection error

Output:

```
output/point_cloud.ply
```

---

### 3. Semantic Annotation

* Uses known world position of VGA socket
* Back-projects into image space
* Generates bounding boxes for:

  * `vga_socket`
  * `ethernet_socket`
  * `power_socket`

---

### 4. 3D Pose Estimation

For each object:

1. Sample grid points inside ROI
2. Triangulate points using multiple views
3. Apply filtering:

   * Reprojection error filtering
   * Percentile filtering
   * MAD-based outlier removal
4. Fit OBB using PCA

---

### 5. Output

Final results are saved as:

```
output/answers.json
```

Format:

```json
{
  "entity": "vga_socket",
  "obb": {
    "center": [...],
    "extent": [...],
    "rotation": [...]
  }
}
```

---

### 6. Validation

* Projects 3D OBB back to images
* Compares predicted vs ground-truth (VGA socket)

---

##  How to Run

### Full pipeline

```bash
python -m src.run_pipeline --grid-points 800
```

### Skip reconstruction (faster)

```bash
python -m src.run_pipeline --skip-reconstruction
```

### Validate only

```bash
python -m src.run_pipeline --validate-only
```

---

##  Results

Example output:

```
Center error: ~0.4 mm
```

Predicted extents (VGA):

```
[0.023, 0.0087, 0.006]
```

Ground truth:

```
[0.0354, 0.0118, 0.0061]
```

✔ Accurate center estimation
✔ Stable multi-view reconstruction
✔ Reasonable bounding box estimation

---

##  Key Techniques Used

* SIFT feature extraction
* Multi-view triangulation (DLT)
* Reprojection error filtering
* Median Absolute Deviation (MAD)
* Principal Component Analysis (PCA)
* Percentile-based bounding box estimation

---

## Limitations

* Bounding boxes depend on ROI quality
* PCA may slightly underestimate object size
* Sensitive to annotation accuracy

---

## 📁 Project Structure

```
src/
 ├── run_pipeline.py
 ├── reconstruction.py
 ├── pose_estimation.py
 ├── semantic.py
 ├── data_loader.py
 ├── utils.py
 └── config.py

output/
 ├── answers.json
 ├── point_cloud.ply
 └── detections/
```

---

## Conclusion

The pipeline successfully reconstructs objects in 3D and estimates their spatial pose with high accuracy. Multi-view geometry combined with semantic information enables reliable reconstruction even with sparse data.

---

##  Author
M Yashwanth
M Jashwanth


