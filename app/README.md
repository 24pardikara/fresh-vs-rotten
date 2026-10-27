---
title: Fresh vs Rotten Fruit Classifier
emoji: 🍎
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Fresh vs Rotten Fruit Classifier

A MobileNetV3 image classifier that labels a fruit photo as **Fresh** or **Rotten**
across Apple, Banana, and Orange (6 classes), with a confidence score.

Deep Learning Lab capstone — Ramdeobaba University.

## This folder IS the Hugging Face Space

To deploy, the Space repo must contain exactly these files at its root:

```
app.py
model_utils.py
requirements.txt
README.md            (this file — the YAML header above configures the Space)
models/mobilenetv3_fresh_rotten.pt
models/labels.json
```

See the project root `README.md` for full deploy steps.
