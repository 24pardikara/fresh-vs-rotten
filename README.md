# Fresh-versus-Rotten Fruit Classification

Deep Learning Lab Capstone — Ramdeobaba University (2026–27)
Team: Priya Keshwani (38) · Pushkar Wankhede (39) · Aastha Pardikar (02)

A vision system that classifies fruit photos into **6 classes** —
Apple / Banana / Orange × **Fresh / Rotten** — and serves predictions through a
web app with a confidence score. A baseline CNN is trained alongside the
proposed **MobileNetV3** so the two can be compared on accuracy and speed.

```
Image → preprocess & augment → MobileNetV3 (transfer learning) → Fresh/Rotten + confidence
```

### 🔗 Live app
**https://fresh-vs-rotten-frayru3aapjtqmoieynnf3.streamlit.app/**
(deployed on Streamlit Community Cloud)

---

## Results (test set)

| Metric | Baseline CNN | **MobileNetV3** |
|---|---|---|
| Accuracy | 0.803 | **0.964** |
| Precision (macro) | 0.855 | **0.966** |
| Recall (macro) | 0.803 | **0.964** |
| F1 (macro) | 0.807 | **0.964** |
| Inference latency | 28.4 ms | **12.6 ms** |
| Throughput | 35.3 FPS | **79.4 FPS** |

MobileNetV3 is both **more accurate (+16 pts)** and **~2× faster** — exactly the
trade-off argued in the proposal. Full per-class numbers and confusion matrices
are in [`reports/RESULTS.md`](reports/RESULTS.md) and
[`reports/confusion_matrices.png`](reports/confusion_matrices.png).

---

## How it was actually built

**Dataset (no Kaggle login needed).** The classic Kaggle set requires an API
token, so instead we pulled the 6 target classes directly from the larger
`Densu341/Fresh-rotten-fruit` mirror on the Hugging Face Hub — reading the
archive's directory remotely and fetching only our classes via HTTP range
requests (see [`src/prep_data.py`](src/prep_data.py)). Result: a balanced
**2,400 images (400/class)**.

**Training (local CPU).** Both models trained with a stratified **70 / 15 / 15**
split, Adam + Cross-Entropy, light augmentation (crop, rotation, brightness).
The deployed model is **MobileNetV3-small @ 128px** (chosen for fast CPU/free-tier
inference). See [`src/train.py`](src/train.py). A full-size GPU run
(MobileNetV3-large @ 224px) is available via the Colab notebook below.

**App.** A Streamlit app ([`app/app.py`](app/app.py)) with image upload **and**
webcam capture, showing predicted class, confidence, per-class probability bars,
and inference latency.

**Deployment.** Hosted free on **Streamlit Community Cloud** from this GitHub
repo. (Note: Hugging Face Spaces now requires a PRO subscription to run
server-based Gradio/Docker apps, so Streamlit Cloud was used instead.)

---

## Project layout

```
fresh-vs-rotten/
├── notebooks/Fresh_vs_Rotten_Training.ipynb   # full-size GPU run (Colab)
├── src/prep_data.py                           # dataset builder (HF Hub, no Kaggle login)
├── src/train.py                               # local training: baseline CNN + MobileNetV3
├── app/app.py                                 # Streamlit app (upload + webcam + confidence)
├── app/model_utils.py                         # model load + inference
├── app/models/                                # trained weights + labels.json
├── reports/                                   # metrics.json, RESULTS.md, confusion matrices
├── deploy_to_hf.py                            # optional Hugging Face deploy helper
└── requirements.txt                           # app + training deps (CPU PyTorch)
```

---

## Reproduce locally

```bash
# 1. install deps
pip install -r requirements.txt

# 2. (re)build the dataset — downloads only the 6 classes, ~2400 images
python src/prep_data.py --per-class 400

# 3. train both models + evaluate (writes app/models/ and reports/)
python src/train.py --arch mobilenet_v3_small --img 128 --cnn-epochs 4 --mnet-epochs 4

# 4. run the app
python -m streamlit run app/app.py
```

For higher final numbers, run `notebooks/Fresh_vs_Rotten_Training.ipynb` in
Google Colab on a GPU (MobileNetV3-large @ 224px) and copy the resulting
`mobilenetv3_fresh_rotten.pt` + `labels.json` into `app/models/`.

---

## Validate on your own fruit (report task)
Photograph fresh fruit, then the same fruit as it naturally overripens over a
few days (with safe handling/disposal), and run those photos through the app to
measure real-world accuracy and collect failure cases.

## Model & evaluation notes
- **Baseline:** 4-block custom CNN trained from scratch.
- **Proposed:** MobileNetV3, ImageNet-pretrained, fine-tuned with a 6-class head;
  Adam + Cross-Entropy.
- **Metrics for both:** accuracy, macro precision/recall/F1, per-class report,
  confusion matrix, single-image latency (ms) and FPS.
