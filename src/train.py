"""CPU-friendly local training for the Fresh-vs-Rotten capstone.

Trains a baseline CNN and a MobileNetV3 (transfer learning) on the images in
`data/raw/<class>/`, does a stratified 70/15/15 split, evaluates both with the
full metric suite (accuracy, precision, recall, F1, per-class, confusion matrix,
latency, FPS), and saves artifacts into `app/models/` and `reports/`.

Defaults are scoped for CPU. For the full GPU run use the Colab notebook.

Usage:  python src/train.py --arch mobilenet_v3_small --img 160 --cnn-epochs 8 --mnet-epochs 5
"""
from __future__ import annotations
import argparse, glob, json, os, random, time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

CLASSES = ["freshapples", "freshbanana", "freshoranges",
           "rottenapples", "rottenbanana", "rottenoranges"]
PRETTY = {"freshapples": "Fresh Apple", "rottenapples": "Rotten Apple",
          "freshbanana": "Fresh Banana", "rottenbanana": "Rotten Banana",
          "freshoranges": "Fresh Orange", "rottenoranges": "Rotten Orange"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- data
class FruitDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label


def gather_split(data_root, seed=42):
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    by_class = defaultdict(list)
    for c in CLASSES:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for fp in glob.glob(os.path.join(data_root, c, ext)):
                by_class[cls_idx[c]].append(fp)
    rng = random.Random(seed)
    tr, va, te = [], [], []
    for lab, paths in by_class.items():
        rng.shuffle(paths)
        n = len(paths)
        n_tr, n_va = int(0.70 * n), int(0.15 * n)
        tr += [(p, lab) for p in paths[:n_tr]]
        va += [(p, lab) for p in paths[n_tr:n_tr + n_va]]
        te += [(p, lab) for p in paths[n_tr + n_va:]]
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te


# ---------------------------------------------------------------- models
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64),
                                      block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.4),
                                        nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_mobilenet(arch, num_classes, pretrained=True):
    if arch == "mobilenet_v3_large":
        w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v3_large(weights=w)
    else:
        arch = "mobilenet_v3_small"
        w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v3_small(weights=w)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m, arch


# ---------------------------------------------------------------- train/eval
def train_model(model, train_dl, val_dl, epochs, lr, device, name):
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train(); run = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            run += loss.item() * x.size(0)
        tr_loss = run / len(train_dl.dataset)
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for x, y in val_dl:
                preds += model(x.to(device)).argmax(1).cpu().tolist(); gts += y.tolist()
        acc = accuracy_score(gts, preds)
        print(f"[{name}] epoch {ep:02d}/{epochs}  train_loss={tr_loss:.4f}  val_acc={acc:.4f}",
              flush=True)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    print(f"[{name}] best val_acc = {best_acc:.4f}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, dl, device, names):
    model.eval(); preds, gts = [], []
    for x, y in dl:
        preds += model(x.to(device)).argmax(1).cpu().tolist(); gts += y.tolist()
    acc = accuracy_score(gts, preds)
    p, r, f1, _ = precision_recall_fscore_support(gts, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1,
            "report": classification_report(gts, preds, target_names=names, zero_division=0),
            "cm": confusion_matrix(gts, preds).tolist()}


@torch.no_grad()
def latency(model, device, img, n=50, warmup=5):
    model.eval()
    d = torch.randn(1, 3, img, img, device=device)
    for _ in range(warmup):
        model(d)
    t0 = time.time()
    for _ in range(n):
        model(d)
    ms = (time.time() - t0) / n * 1000
    return {"latency_ms": ms, "fps": 1000 / ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "raw"))
    ap.add_argument("--arch", default="mobilenet_v3_small",
                    choices=["mobilenet_v3_small", "mobilenet_v3_large"])
    ap.add_argument("--img", type=int, default=160)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cnn-epochs", type=int, default=8)
    ap.add_argument("--mnet-epochs", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    print("Device:", device, "| arch:", args.arch, "| img:", args.img, flush=True)

    tr, va, te = gather_split(args.data)
    print(f"Split -> train {len(tr)}  val {len(va)}  test {len(te)}", flush=True)
    if not tr:
        raise SystemExit(f"No images found in {args.data}. Run src/prep_data.py first.")

    train_tf = T.Compose([
        T.Resize((args.img + 32, args.img + 32)),
        T.RandomResizedCrop(args.img, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(), T.RandomRotation(20),
        T.ColorJitter(brightness=0.3, contrast=0.2),
        T.ToTensor(), T.Normalize(MEAN, STD)])
    eval_tf = T.Compose([
        T.Resize((args.img + 32, args.img + 32)), T.CenterCrop(args.img),
        T.ToTensor(), T.Normalize(MEAN, STD)])

    train_dl = DataLoader(FruitDataset(tr, train_tf), batch_size=args.batch,
                          shuffle=True, num_workers=0)
    val_dl = DataLoader(FruitDataset(va, eval_tf), batch_size=args.batch, num_workers=0)
    test_dl = DataLoader(FruitDataset(te, eval_tf), batch_size=args.batch, num_workers=0)

    pretty = [PRETTY[c] for c in CLASSES]

    # baseline CNN
    cnn = BaselineCNN(len(CLASSES))
    cnn = train_model(cnn, train_dl, val_dl, args.cnn_epochs, 1e-3, device, "BaselineCNN")
    cnn_res = evaluate(cnn, test_dl, device, pretty)
    cnn_lat = latency(cnn, device, args.img)

    # MobileNetV3 transfer learning
    mnet, arch = build_mobilenet(args.arch, len(CLASSES), pretrained=True)
    mnet = train_model(mnet, train_dl, val_dl, args.mnet_epochs, 3e-4, device, "MobileNetV3")
    mnet_res = evaluate(mnet, test_dl, device, pretty)
    mnet_lat = latency(mnet, device, args.img)

    print("\n===== Baseline CNN =====\n" + cnn_res["report"])
    print(f"Latency {cnn_lat['latency_ms']:.1f} ms | FPS {cnn_lat['fps']:.1f}")
    print("\n===== MobileNetV3 =====\n" + mnet_res["report"])
    print(f"Latency {mnet_lat['latency_ms']:.1f} ms | FPS {mnet_lat['fps']:.1f}")

    print("\n============ COMPARISON ============")
    print(f"{'metric':<14}{'CNN':>12}{'MobileNetV3':>14}")
    for k in ["accuracy", "precision", "recall", "f1"]:
        print(f"{k:<14}{cnn_res[k]:>12.4f}{mnet_res[k]:>14.4f}")
    print(f"{'latency_ms':<14}{cnn_lat['latency_ms']:>12.1f}{mnet_lat['latency_ms']:>14.1f}")
    print(f"{'fps':<14}{cnn_lat['fps']:>12.1f}{mnet_lat['fps']:>14.1f}")

    # ---- save artifacts
    os.makedirs(os.path.join("app", "models"), exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    torch.save(mnet.state_dict(), os.path.join("app", "models", "mobilenetv3_fresh_rotten.pt"))
    torch.save(cnn.state_dict(), os.path.join("reports", "baseline_cnn.pt"))
    with open(os.path.join("app", "models", "labels.json"), "w") as f:
        json.dump({"classes": CLASSES, "pretty": pretty, "arch": arch,
                   "img_size": args.img, "mean": MEAN, "std": STD}, f, indent=2)
    metrics = {
        "baseline_cnn": {k: cnn_res[k] for k in ["accuracy", "precision", "recall", "f1"]} | cnn_lat,
        "mobilenetv3": {k: mnet_res[k] for k in ["accuracy", "precision", "recall", "f1"]} | mnet_lat,
        "arch": arch, "img_size": args.img,
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "confusion_matrix": {"baseline_cnn": cnn_res["cm"], "mobilenetv3": mnet_res["cm"]},
        "classes": pretty,
    }
    with open(os.path.join("reports", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # confusion matrix figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def plot_cm(cm, names, title, ax):
            cm = np.array(cm)
            ax.imshow(cm, cmap="Blues")
            ax.set_title(title)
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        plot_cm(cnn_res["cm"], pretty, "Baseline CNN", axes[0])
        plot_cm(mnet_res["cm"], pretty, "MobileNetV3", axes[1])
        plt.tight_layout(); plt.savefig(os.path.join("reports", "confusion_matrices.png"), dpi=120)
        print("Saved reports/confusion_matrices.png")
    except Exception as e:
        print("Skipped confusion-matrix plot:", e)

    print("\nArtifacts written to app/models/ and reports/. Done.", flush=True)


if __name__ == "__main__":
    main()
