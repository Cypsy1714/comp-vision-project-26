import argparse
import copy
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights


# ---------------------------------------------------------------------------
# device + transforms
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_device(cfg) -> torch.device:
    d = cfg.get("device", "auto")
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def make_transform(cfg) -> transforms.Compose:
    """Clean deterministic transform — used for val and scoring."""
    n = cfg["image_size"]
    return transforms.Compose([
        transforms.Resize((n, n)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def _make_train_transform(cfg) -> transforms.Compose:
    """Augmented transform — training only, never applied to val or scoring."""
    n = cfg["image_size"]
    return transforms.Compose([
        transforms.RandomResizedCrop(n, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class CropDataset(Dataset):
    """
    Reads crops/{stem}.jpg, labels from labels.csv.
    Rows with label == -1 are excluded (handled by the meta stage).
    Pass transform=None only when you need the index for splitting — never
    call __getitem__ on a None-transform instance.
    """

    def __init__(self, crops_dir: Path, labels_csv: Path, transform):
        with open(labels_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        self.samples = [
            (crops_dir / r["filename"], int(r["label"]))
            for r in rows
            if int(r["label"]) != -1 and (crops_dir / r["filename"]).exists()
        ]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def labels(self) -> list[int]:
        return [lbl for _, lbl in self.samples]


def _stratified_split(dataset: CropDataset, val_frac: float, seed: int):
    """
    80 / 20 stratified split.  Classes with < 2 images go entirely to train
    so we never crash on tiny classes.
    Returns (train_indices, val_indices).
    """
    rng = random.Random(seed)

    by_class: dict[int, list[int]] = {}
    for i, lbl in enumerate(dataset.labels):
        by_class.setdefault(lbl, []).append(i)

    train_idx, val_idx = [], []
    print("[cnn] per-class split:")
    for cls in sorted(by_class):
        idxs = by_class[cls]
        rng.shuffle(idxs)
        n_val = math.ceil(len(idxs) * val_frac) if len(idxs) >= 2 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
        print(f"  class {cls:2d}: {len(idxs):3d} total → train {len(idxs) - n_val}  val {n_val}")

    return train_idx, val_idx


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def build_model(cfg) -> nn.Module:
    pretrained = cfg["cnn"]["pretrained"]
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, cfg["num_classes"])
    return model


# ---------------------------------------------------------------------------
# macro-F1 (sklearn if present, else manual)
# ---------------------------------------------------------------------------

def _macro_f1(preds: list[int], targets: list[int], num_classes: int) -> float:
    try:
        from sklearn.metrics import f1_score
        return float(f1_score(targets, preds, average="macro", zero_division=0))
    except ImportError:
        pass
    f1s = []
    for c in range(num_classes):
        tp = sum(p == c and t == c for p, t in zip(preds, targets))
        fp = sum(p == c and t != c for p, t in zip(preds, targets))
        fn = sum(p != c and t == c for p, t in zip(preds, targets))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return sum(f1s) / len(f1s)


# ---------------------------------------------------------------------------
# train / eval helpers
# ---------------------------------------------------------------------------

def _train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


def _eval_epoch(model, loader, criterion, device, num_classes: int):
    model.eval()
    total_loss, correct = 0.0, 0
    all_preds: list[int] = []
    all_targets: list[int] = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            total_loss += criterion(logits, labels).item() * len(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())
    n = len(loader.dataset)
    return total_loss / n, correct / n, _macro_f1(all_preds, all_targets, num_classes)


# ---------------------------------------------------------------------------
# train()  — callable from __main__ via --train / --scratch
# ---------------------------------------------------------------------------

def train(cfg, weights_path: Path) -> nn.Module:
    device = get_device(cfg)
    print(f"[cnn] device={device}  weights → {weights_path}")

    c           = cfg["cnn"]
    labels_csv  = Path(cfg["paths"]["input"]) / "images" / "labels.csv"
    crops_dir   = Path(cfg["paths"]["crops"])
    num_classes = cfg["num_classes"]

    # build index dataset (transform=None; __getitem__ never called on this)
    index_ds = CropDataset(crops_dir, labels_csv, transform=None)
    train_idx, val_idx = _stratified_split(index_ds, val_frac=0.2, seed=cfg["seed"])

    # class weights from training labels only
    train_labels = [index_ds.labels[i] for i in train_idx]
    counts       = [train_labels.count(cl) for cl in range(num_classes)]
    N            = len(train_labels)
    cls_weights  = torch.tensor(
        [N / (num_classes * max(cnt, 1)) for cnt in counts], dtype=torch.float32
    ).to(device)

    # two datasets with appropriate transforms sharing the same CSV
    train_ds = CropDataset(crops_dir, labels_csv, transform=_make_train_transform(cfg))
    val_ds   = CropDataset(crops_dir, labels_csv, transform=make_transform(cfg))

    train_loader = DataLoader(
        Subset(train_ds, train_idx), batch_size=c["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx), batch_size=c["batch_size"], shuffle=False, num_workers=0
    )

    criterion = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=0.1)
    model     = build_model(cfg).to(device)

    # ── Phase A: head only ──────────────────────────────────────────────────
    for p in model.features.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=c["head"]["lr"],
    )
    print(f"\n[cnn] Phase A — head only  {c['head']['epochs']} epochs  lr={c['head']['lr']}")
    for epoch in range(c["head"]["epochs"]):
        tr_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vl_f1 = _eval_epoch(model, val_loader, criterion, device, num_classes)
        print(
            f"  A {epoch+1:02d}/{c['head']['epochs']}"
            f"  train_loss={tr_loss:.4f}"
            f"  val_loss={vl_loss:.4f}"
            f"  val_acc={vl_acc:.3f}"
            f"  val_f1={vl_f1:.3f}"
        )

    # ── Phase B: full fine-tune with early stopping ──────────────────────────
    for p in model.features.parameters():
        p.requires_grad_(True)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=c["full"]["lr"])
    patience   = 8
    best_f1    = -1.0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    print(f"\n[cnn] Phase B — full fine-tune  up to {c['full']['epochs']} epochs  lr={c['full']['lr']}  patience={patience}")
    for epoch in range(c["full"]["epochs"]):
        tr_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vl_f1 = _eval_epoch(model, val_loader, criterion, device, num_classes)

        marker = ""
        if vl_f1 > best_f1:
            best_f1    = vl_f1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            marker     = " *"
        else:
            no_improve += 1

        print(
            f"  B {epoch+1:02d}/{c['full']['epochs']}"
            f"  train_loss={tr_loss:.4f}"
            f"  val_loss={vl_loss:.4f}"
            f"  val_acc={vl_acc:.3f}"
            f"  val_f1={vl_f1:.3f}{marker}"
        )

        if no_improve >= patience:
            print(f"  [cnn] early stop — no val_f1 gain for {patience} consecutive epochs")
            break

    model.load_state_dict(best_state)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f"[cnn] saved best weights → {weights_path}  (best val_f1={best_f1:.3f})")
    return model


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _score(cfg, model: nn.Module, device: torch.device):
    views_dir  = Path(cfg["paths"]["views"])
    scores_dir = Path(cfg["paths"]["scores"])
    scores_dir.mkdir(parents=True, exist_ok=True)

    tf      = make_transform(cfg)
    active  = cfg["views"]["active"]
    softmax = nn.Softmax(dim=1)
    stems   = sorted({p.name.split("__")[0] for p in views_dir.glob("*.jpg")})

    model.eval()
    with torch.no_grad():
        for stem in stems:
            imgs = []
            for v in active:
                fp = views_dir / f"{stem}__{v}.jpg"
                imgs.append(
                    tf(Image.open(fp).convert("RGB"))
                    if fp.exists()
                    else torch.zeros(3, cfg["image_size"], cfg["image_size"])
                )
            batch = torch.stack(imgs).to(device)                         # (K, 3, H, W)
            probs = softmax(model(batch)).cpu().numpy().astype("float32") # (K, 20)
            np.save(scores_dir / f"{stem}.npy", probs)

    print(f"[cnn] scores saved → {scores_dir}  ({len(stems)} stems)")


# ---------------------------------------------------------------------------
# public entry point — called by main.py
# ---------------------------------------------------------------------------

def run(cfg):
    device = get_device(cfg)
    print(f"[cnn] device={device}")

    model        = build_model(cfg).to(device)
    weights_path = Path("weights/cnn.pt")

    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[cnn] loaded weights from {weights_path}")
    else:
        print(f"[cnn] WARNING: {weights_path} not found — scoring with untrained head (pretrained backbone only)")

    _score(cfg, model, device)


# ---------------------------------------------------------------------------
# CLI  (python src/cnn.py [--train] [--scratch])
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml"))

    parser = argparse.ArgumentParser(description="Stage C — CNN train / score")
    parser.add_argument("--train",   action="store_true", help="train and save weights/cnn.pt")
    parser.add_argument("--scratch", action="store_true", help="train from scratch (pretrained=False), save weights/cnn_scratch.pt")
    args = parser.parse_args()

    if args.scratch:
        scratch_cfg = copy.deepcopy(cfg)
        scratch_cfg["cnn"]["pretrained"] = False
        train(scratch_cfg, Path("weights/cnn_scratch.pt"))
    elif args.train:
        train(cfg, Path("weights/cnn.pt"))
    else:
        run(cfg)
