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

## Project layout

```
fresh-vs-rotten/
├── notebooks/
│   └── Fresh_vs_Rotten_Training.ipynb   # ← train here (Google Colab, free GPU)
├── app/                                 # ← Streamlit app = the Hugging Face Space
│   ├── app.py                           #    upload + webcam + confidence UI
│   ├── model_utils.py                   #    model load + inference
│   ├── requirements.txt
│   ├── README.md                        #    HF Space config (YAML header)
│   └── models/                          #    trained weights go here
├── reports/                             # confusion matrices, metrics (from Colab)
└── README.md                            # this file
```

---

## Pipeline: Train → Test → Validate

### 1. Train & test (Google Colab — needs your Kaggle token)
1. Open `notebooks/Fresh_vs_Rotten_Training.ipynb` in Google Colab
   (colab.research.google.com → File → Upload notebook).
2. `Runtime → Change runtime type → T4 GPU`.
3. **Kaggle token:** kaggle.com → Settings → API → *Create New Token* → saves
   `kaggle.json`. Run the download cell and upload that file when prompted.
4. Run all cells. The notebook will:
   - download the Kaggle dataset (~13.6k images, 6 classes);
   - make a stratified **70 / 15 / 15** train/val/test split;
   - augment (crop, rotation, brightness), train the **baseline CNN** and
     **MobileNetV3**, and evaluate both on the test set;
   - report accuracy, precision, recall, F1, per-class scores, confusion
     matrices, and **inference latency + FPS**;
   - save and download `artifacts.zip`.
5. Unzip `artifacts.zip` and copy **`mobilenetv3_fresh_rotten.pt`** and
   **`labels.json`** into `app/models/`. Put `confusion_matrices.png` and
   `metrics.json` in `reports/` for the write-up.

### 2. Run the app locally (this PC)
```bash
cd fresh-vs-rotten/app
python -m venv .venv && .venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```
Open the URL it prints. Upload a fruit photo or use the webcam tab; the app
shows the predicted class, confidence, full probability bars, and latency/FPS.

### 3. Validate on your own fruit (your team's physical task)
Photograph fresh fruit, then the same fruit as it naturally overripens over a
few days (follow the safe-handling/disposal note in the proposal). Run those
photos through the app (or the notebook's eval cell) to measure real-world
accuracy and collect failure cases for the report.

---

## Deploy to Hugging Face Spaces (free)

> Resource gate: this needs a **free Hugging Face account** — tell me when it's
> created and I'll walk the exact commands.

1. Create an account at huggingface.co, then a new **Space**:
   New → Space → SDK **Streamlit** → name it (e.g. `fresh-vs-rotten`).
2. The Space is a git repo. Push the **contents of `app/`** (not the folder) to
   its root — `app.py`, `model_utils.py`, `requirements.txt`, `README.md`, and
   `models/mobilenetv3_fresh_rotten.pt` + `models/labels.json`:
   ```bash
   git clone https://huggingface.co/spaces/<user>/fresh-vs-rotten
   cp -r app/* fresh-vs-rotten/        # copy app contents into the clone
   cd fresh-vs-rotten
   git add . && git commit -m "Deploy fresh-vs-rotten classifier"
   git push
   ```
   (The `.pt` file is a few MB — fine for a normal git push. If git complains
   about size, enable Git LFS for `*.pt`.)
3. The Space builds and goes live at
   `https://huggingface.co/spaces/<user>/fresh-vs-rotten`.

---

## Model & evaluation notes
- **Baseline:** 4-block custom CNN trained from scratch.
- **Proposed:** MobileNetV3-Large, ImageNet-pretrained, fine-tuned with a
  6-class head; Adam + Cross-Entropy, as specified in the proposal.
- **Metrics reported for both:** accuracy, macro precision/recall/F1, per-class
  report, confusion matrix, single-image latency (ms) and FPS.
