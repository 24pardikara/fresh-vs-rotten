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

## Error analysis (MobileNetV3)

All 13 of MobileNetV3's errors on the 360-image test set, read straight off the
confusion matrix in `metrics.json`:

| True | Predicted | Count | Wrong in |
|---|---|---|---|
| Rotten Orange | Fresh Orange | 6 | freshness only |
| Rotten Apple | Fresh Apple | 4 | freshness only |
| Fresh Apple | Rotten Apple | 2 | freshness only |
| Rotten Apple | Fresh Orange | 1 | both |

There is **no** Rotten Apple ↔ Rotten Orange confusion — an earlier draft of this
file claimed there was, which the matrix does not support.

Because each label is *species × freshness*, the matrix can be collapsed along
either axis to see which half of the task the errors belong to:

| | Baseline CNN | MobileNetV3 |
|---|---|---|
| Overall accuracy | 0.8028 | **0.9639** |
| Species accuracy (apple/banana/orange) | 0.8861 | **0.9972** |
| Freshness accuracy (fresh/rotten) | 0.8861 | **0.9639** |
| Freshness-only errors | 30 | 12 |
| Species-involving errors | 41 | 1 |

MobileNetV3 identifies the fruit correctly in 359 of 360 images, and 12 of its 13
mistakes are pure freshness errors on a correctly recognised fruit. Object
recognition is effectively solved; **every point of remaining headroom sits on the
fresh/rotten boundary** — which is the real "failure case" to discuss, and the
argument for ordinal freshness labels, a borderline class, or a two-head model.

The baseline shows no such separation (species and freshness accuracy are both
0.8861) and instead collapses onto one attractor class: the Rotten Apple column
receives 106 of 360 predictions, including 25 of the 60 fresh apples.

See `confusion_matrices.png` for both models side by side, and `metrics.json`
for the raw numbers.

## Artifacts
- `app/models/mobilenetv3_fresh_rotten.pt` — trained MobileNetV3 (served by the app)
- `app/models/labels.json` — classes + preprocessing (arch, img size, normalization)
- `reports/baseline_cnn.pt` — baseline model
- `reports/metrics.json`, `reports/confusion_matrices.png`

