# Robust Lane Detection Under Varied Lighting Conditions


**Authors:** Nishant Suresh · Kateryna Lysytsyna · Sagar Kaistha

---

## Running Inference: 

### Run Inference on the Logistic Regression Model
Notebook link: https://www.kaggle.com/code/nishantsuresh1/logistic-regression-nishant-sagar-kateryna
Run Sections 1 and 2: Import and helper functions.
Skip Sections 3, 5, 6, and 7.
Run Sections 8 and 9.

Required dataset inputs (must be attached to the notebook):
CULane dataset: manideep1108/culane
Saved model weights: nishantsuresh1/lr-hog/lr_best.pkl and nishantsuresh1/lr-hog/scaler.pkl

### Run Inference on the Lightweight CNN Model
A ready-to-run inference cell is included at the bottom of `lightweight-cnn.ipynb`. It loads the saved checkpoint and runs the model on `road.jpg` (included in the repo root), producing a binary lane mask and a green overlay visualization.

1. Open `lightweight-cnn.ipynb`
2. Set `IMAGE_PATH` and `WEIGHTS_PATH` in the config block (defaults already point to `road.jpg` and `lightweight_cnn_best.pth`)
3. Run the inference cell — outputs are saved as `lane_mask.png` and `lane_overlay.png`

### Run Inference on the U-Net Model
Notebook link: https://www.kaggle.com/code/katlysytsyna/u-net-nishantsuresh-katerynalysytsyna-sagarkaitha/edit
Cell 1 — Preprocessing: Loads the CULane dataset, defines preprocessing functions, builds image-mask pairs, and creates the DataLoader.
Cell 2 — Model Definition: Defines the DoubleConv block and U-Net architecture (31M parameters).
Cell 3 — Validation Set: Loads the CULane validation split and creates the validation DataLoader.
Cell 4 — Loss Function: Defines the combined BCE + Dice loss.
Cell 5 — Load Saved Model: Loads the pretrained weights from the best checkpoint (epoch 4). Path: /kaggle/input/datasets/katlysytsyna/unet-model/unet_best.pth
Cell 6 — Testing & Visualization: Runs inference on all 9 CULane test scenarios (normal, crowd, highlight, shadow, no line, arrow, curve, crossroad, night), computes IoU/F1/Precision/Recall per scenario, and displays prediction visualizations.

Skip: Between Cells 4 and 5: The training cell (NUM_EPOCHS = 7) — the model is already trained and saved.
Required dataset inputs (must be attached to the notebook):
CULane dataset: manideep1108/culane
Saved model weights: katlysytsyna/unet-model

---

## Overview

Lane detection is one of the most critical components in advanced driver assistance systems (ADAS) and autonomous driving. Lane departures account for approximately **51% of all fatal car crashes** in the United States, resulting in an estimated **19,000 deaths** annually (Federal Highway Administration, 2023).

This project frames lane detection as **pixel-wise binary semantic segmentation** — each image pixel is classified as either `lane` or `background`. Three complementary models of increasing complexity are trained and evaluated on the [CULane dataset](https://xingangpan.github.io/projects/CULane.html) across nine diverse driving scenarios.

---

## Dataset: CULane

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

## Challenges

- **Low contrast** at night makes lane markings difficult to distinguish
- **Shadow boundaries** create uneven illumination across the road surface
- **Glare** from sunlight or headlights washes out lane markings
- **Severe class imbalance** — only ~0.5–5% of pixels are lane pixels per image

---

## Models

All models perform **pixel-wise binary classification** (lane vs. background).

| Model | Type | Input | Library | Mean IoU | Mean F1 |
|-------|------|-------|---------|----------|---------|
| Logistic Regression | Baseline | 326 HOG features from 32×32 patch | scikit-learn | **0.500** | **0.657** |
| Lightweight CNN | Shallow DNN | 3×256×512 RGB image | PyTorch | 0.207 | 0.329 |
| U-Net | Deep DNN | 3×256×512 RGB image | PyTorch | 0.365 | 0.494 |

### Why Semantic Segmentation?
- Naturally handles complex geometry (curves, turns, intersections)
- Pixel-level precision for lane boundary detection
- Automatic feature learning
- Robust to partial occlusions
- U-Net's encoder-decoder architecture with skip connections is well-suited for this task

---

## Methodology

### Preprocessing
- Resize images to **512×256** to reduce computational cost
- Normalize pixel values to **[0, 1]**
- Convert ground-truth masks to **binary** (0 = background, 1 = lane)

### Data Augmentation (DNN models only)
- Random horizontal flips (p = 0.5)
- Random brightness adjustment (factor 0.6–1.4)
- Random contrast adjustment (factor 0.7–1.3)
- Random Gaussian noise (σ = 0.02)

### Loss Functions
- **Logistic Regression:** Balanced patch sampling (equal lane/background patches); L2 regularisation tuned via cross-validation (best C = 10⁻⁴)
- **Lightweight CNN:** Combined loss — 0.5 · BCE (pos_weight=10) + 0.5 · Dice
- **U-Net:** Combined loss — BCE (unweighted) + Dice

### Training
- **Lightweight CNN:** Adam (lr=1e-3), batch size 32, 20 epochs, ReduceLROnPlateau, AMP mixed precision
- **U-Net:** Adam (lr=3e-4, weight decay 1e-4), batch size 8, 7 epochs, ReduceLROnPlateau, AMP mixed precision on Tesla T4

### Inference
1. Model outputs per-pixel probability maps
2. Apply sigmoid threshold (0.5) to produce binary lane masks
3. No post-processing applied

---

## Results

### Logistic Regression

| Scenario | IoU | F1 |
|----------|-----|----|
| Normal | 0.571 | 0.727 |
| Arrow | 0.580 | 0.734 |
| Curve | 0.545 | 0.706 |
| Highlight | 0.530 | 0.693 |
| Shadow | 0.517 | 0.681 |
| No Line | 0.437 | 0.608 |
| Crowded | 0.412 | 0.583 |
| Night | 0.350 | 0.518 |
| Cross | FPR = 38.65% | — |
| **Mean** | **0.500** | **0.657** |

### Lightweight CNN

| Scenario | IoU | F1 | F1−IoU |
|----------|-----|-----|--------|
| Normal | 0.32 | 0.48 | 0.155 |
| Curve | 0.27 | 0.41 | 0.145 |
| Intersection | 0.27 | 0.41 | 0.145 |
| Crowded | 0.20 | 0.32 | 0.120 |
| Arrow | 0.17 | 0.28 | 0.150 |
| Highlight | 0.17 | 0.28 | 0.110 |
| Shadow | 0.17 | 0.28 | 0.110 |
| No Line | 0.16 | 0.27 | 0.110 |
| Night | 0.13 | 0.23 | 0.095 |
| **Mean** | **0.207** | **0.329** | 0.127 |

Training curves and per-scenario evaluation charts are saved in `lightweight_CNN_results/`.

### U-Net

| Scenario | IoU | Prec | Rec | F1 |
|----------|-----|------|-----|----|
| Normal | 0.614 | 0.777 | 0.743 | 0.756 |
| Arrow | 0.543 | 0.772 | 0.645 | 0.698 |
| Crowd | 0.469 | 0.704 | 0.563 | 0.602 |
| Curve | 0.398 | 0.682 | 0.484 | 0.552 |
| Shadow | 0.328 | 0.719 | 0.376 | 0.483 |
| Highlight | 0.313 | 0.669 | 0.372 | 0.466 |
| No Line | 0.148 | 0.509 | 0.172 | 0.239 |
| Night | 0.107 | 0.359 | 0.119 | 0.157 |
| Crossroad | 0.000 | 0.000 | 0.000 | 0.000 |
| **Mean** | **0.365** | **0.577** | **0.386** | **0.494** |

> U-Net reaches **96% of published state-of-the-art** on the Normal scenario (IoU 0.614 vs. SOTA 0.64).

---

## Evaluation Metrics

- **IoU (Intersection over Union):** `TP / (TP + FP + FN)` — penalises both false positives and false negatives equally. Pixel accuracy is discarded as a metric due to severe class imbalance (all-background prediction yields >95% accuracy but IoU = 0).
- **F1 Score:** Harmonic mean of precision and recall. The F1–IoU gap is a useful diagnostic: positive gap indicates recall bias (over-prediction); negative gap indicates precision bias (under-prediction).

---

## Project Structure

```
├── docs/
│   ├── Presentation.pdf
│   └── Proposal.pdf
├── lightweight_CNN_results/
│   ├── scenario_eval.png          # Per-scenario grouped bar chart
│   ├── scenario_results.csv       # Per-scenario IoU & F1 scores
│   ├── training_curves.png        # Loss / IoU / F1 over epochs
│   └── training_history.csv      # Raw training metrics per epoch
├── .gitignore
├── lightweight-cnn.ipynb          # Training, evaluation & inference
├── lightweight_cnn_best.pth       # Best checkpoint (Val IoU = 0.2075, epoch 9)
├── road.jpg                       # Example image for inference
└── README.md
```

---

## References

1. Federal Highway Administration. (2023). [Roadway Departure Safety](https://highways.dot.gov/safety/RwD). U.S. Department of Transportation.
2. Pan, X. et al. (2018). [Spatial as Deep: Spatial CNN for Traffic Scene Understanding](https://arxiv.org/abs/1709.04108). AAAI 2018. *(Introduces CULane dataset and SCNN.)*
3. Tabelini, L. et al. (2021). [Keep your Eyes on the Lane: Real-time Attention-guided Lane Detection](https://arxiv.org/abs/2010.12035). CVPR 2021.
4. Zheng, T. et al. (2022). [CLRNet: Cross Layer Refinement Network for Lane Detection](https://arxiv.org/abs/2203.10350). CVPR 2022.
5. Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). MICCAI 2015.
