# 🚗 Robust Lane Detection Under Varied Lighting Conditions

> Pixel-wise semantic segmentation for autonomous driving safety across diverse real-world scenarios.

**Authors:** Nishant Suresh · Kateryna Lysytsyna · Sagar Kaistha

---

## 📌 Overview

Lane detection is one of the most critical components in advanced driver assistance systems (ADAS) and autonomous driving. Lane departures account for approximately **50% of all fatal car crashes** in the United States, resulting in an estimated **13,000–19,000 deaths** and nearly **35,000 injuries** annually.

This project frames lane detection as **pixel-wise semantic segmentation** — each image pixel is classified as either `lane` or `background`. Three complementary models are trained and evaluated on the [CULane dataset](https://xingangpan.github.io/projects/CULane.html) across nine diverse driving scenarios.

---

## 🗂️ Dataset: CULane

Collected in Beijing by CUHK & SenseTime. Resolution: **1640×590**.

| Split | Frames |
|-------|--------|
| Train | 88,880 |
| Validation | 9,675 |
| Test | 34,680 |
| **Total** | **133,235** |

### Scenarios
| # | Scenario | Description |
|---|----------|-------------|
| 1 | Normal | Daytime driving |
| 2 | Shadow | Trees / overpasses |
| 3 | Curve | Sharp turns |
| 4 | Highlight | Glare / dazzle |
| 5 | Crowded | Occlusions |
| 6 | Arrow | Arrow markings |
| 7 | No Line | Unmarked roads |
| 8 | Night | Low-light conditions |
| 9 | Intersection | Complex areas |

### Annotations
- Per-pixel segmentation labels (`0` = background, `1–4` = lanes)
- Lane existence flags
- Keypoint coordinates (cubic splines)

---

## ⚠️ Challenges

- **Low contrast** at night makes lane markings difficult to distinguish
- **Shadow boundaries** create uneven illumination across the road surface
- **Glare** from sunlight or headlights washes out lane markings
- **Severe class imbalance** — only ~0.5% of pixels are lane pixels

---

## 🧠 Models

All models perform **pixel-wise binary classification** (lane vs. background).

| Model | Type | Input | Library |
|-------|------|-------|---------|
| Logistic Regression | Baseline | 326 HOG features from 32×32 patch | scikit-learn |
| Lightweight CNN | Shallow DNN | 3×256×512 RGB image | PyTorch |
| U-Net | Deep DNN | 3×256×512 RGB image | PyTorch |

### Why Semantic Segmentation?
- Naturally handles complex geometry (curves, turns, intersections)
- Pixel-level precision for lane boundary detection
- Automatic feature learning
- Robust to partial occlusions
- U-Net's encoder-decoder architecture with skip connections is well-suited for this task

---

## ⚙️ Methodology

### Preprocessing
- Resize images to **512×256** to reduce computational cost
- Normalize pixel values to **[0, 1]**
- Convert ground-truth masks to **binary** (0 = background, 1 = lane)

### Training
- Loss functions: **Cross-Entropy** or **Dice Loss** for pixel-wise classification
- Optimizer: Standard gradient-based optimization (PyTorch)

### Inference
1. Model outputs per-pixel probability maps
2. Apply threshold (e.g., **0.5**) to produce binary lane masks
3. Post-process masks to extract lane boundaries for vehicle control

---

## 📊 Evaluation

Evaluation follows the standard CULane benchmark protocol.

- **IoU (Intersection over Union):** Intersection of predicted and ground-truth lanes divided by their union
- **F1 Score:** Computed per-scenario using IoU threshold:
  - `TP`: IoU ≥ 0.5
  - `FP`: IoU < 0.5

Results are reported **per scenario** (daytime, night, shadow, glare, etc.) to assess robustness across real-world driving conditions.

---

## 📁 Project Structure

```
├── data/                  # CULane dataset
├── models/
│   ├── logistic_regression.py
│   ├── cnn.py
│   └── unet.py
├── train.py               # Training script
├── evaluate.py            # Evaluation & metrics
├── preprocess.py          # Data preprocessing utilities
└── README.md
```

---

## 🔗 References

- [CULane Dataset](https://xingangpan.github.io/projects/CULane.html)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)