# Results — Fresh vs Rotten Fruit Classification

Trained locally on CPU (12 cores) on 2,400 images (400/class), stratified
**70 / 15 / 15** split → train 1680 / val 360 / test 360.
MobileNetV3-**small** (transfer learning) vs a baseline CNN from scratch.
Input 128×128, Adam, Cross-Entropy, light augmentation (crop, rotation,
brightness). *(Re-run the Colab notebook at 224px / MobileNetV3-large for even
higher final numbers.)*

## Test-set comparison

| Metric | Baseline CNN | MobileNetV3 |
|---|---|---|
| Accuracy | 0.803 | **0.964** |
| Precision (macro) | 0.855 | **0.966** |
| Recall (macro) | 0.803 | **0.964** |
| F1 (macro) | 0.807 | **0.964** |
| Inference latency | 28.4 ms | **12.6 ms** |
| Throughput | 35.3 FPS | **79.4 FPS** |

MobileNetV3 is both **more accurate (+16 pts)** and **~2× faster** — the exact
trade-off argued in the proposal.

## MobileNetV3 per-class (test)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Fresh Apple | 0.94 | 0.97 | 0.95 |
| Fresh Banana | 1.00 | 1.00 | 1.00 |
| Fresh Orange | 0.90 | 1.00 | 0.94 |
| Rotten Apple | 0.96 | 0.92 | 0.94 |
| Rotten Banana | 1.00 | 1.00 | 1.00 |
| Rotten Orange | 1.00 | 0.90 | 0.95 |

Main confusion: a few Rotten Apple ↔ Rotten Orange mix-ups (visually similar
dark/mottled surfaces) — a good "failure case" to discuss in the report.

See `confusion_matrices.png` for both models side by side, and `metrics.json`
for the raw numbers.

## Artifacts
- `app/models/mobilenetv3_fresh_rotten.pt` — trained MobileNetV3 (served by the app)
- `app/models/labels.json` — classes + preprocessing (arch, img size, normalization)
- `reports/baseline_cnn.pt` — baseline model
- `reports/metrics.json`, `reports/confusion_matrices.png`
