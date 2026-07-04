import argparse
import copy
import csv
import math
import random
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# imagenet normalization, used for every model so inputs are always the same
MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD = np.array([0.229, 0.224, 0.225], dtype="float32")


# pick the compute device from the config ("auto" tries cuda, then mps, then cpu)
def get_device(cfg):
    d = cfg["device"]
    if d != "auto":
        return torch.device(d)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# timm backbone (imagenet weights or random) with our own pooled linear head on top
class TimmModel(nn.Module):
    def __init__(self, backbone, pretrained, num_classes):
        super().__init__()
        self.base = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.base.num_features, num_classes)

    def forward(self, x):
        return self.head(self.pool(self.base.forward_features(x)).flatten(1))

    def freeze_base(self, freeze):
        for p in self.base.parameters():
            p.requires_grad = not freeze


# small cnn built from the config: conv blocks from "channels", dense layers from "hidden"
class CustomCNN(nn.Module):
    def __init__(self, channels, hidden, num_classes):
        super().__init__()
        blocks, c_in = [], 3
        for c in channels:
            blocks += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c
        self.base = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        layers = []
        for h in hidden:
            layers += [nn.Linear(c_in, h), nn.ReLU()]
            c_in = h
        self.head = nn.Sequential(*layers, nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.base(x)).flatten(1))

    def freeze_base(self, freeze):
        for p in self.base.parameters():
            p.requires_grad = not freeze


# build one model from its entry in cnn.models
def build_model(mcfg, num_classes):
    if mcfg["type"] == "timm":
        return TimmModel(mcfg["backbone"], mcfg["pretrained"], num_classes)
    return CustomCNN(mcfg["channels"], mcfg["hidden"], num_classes)


# macro f1: f1 per class, averaged over all classes (same metric as the example project)
def macro_f1(y_true, y_pred, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(f1s) / num_classes


# inverse-frequency class weights, normalized so the mean weight is ~1 (same math as the example)
def class_weights(labels, num_classes):
    counts = torch.zeros(num_classes, dtype=torch.double)
    for y in labels:
        counts[y] += 1
    present = counts > 0
    inv = torch.zeros(num_classes, dtype=torch.double)
    inv[present] = 1.0 / counts[present]
    if present.any():
        inv[present] = inv[present] / inv[present].mean()
    inv[~present] = 1.0
    return inv.float()


# lr schedule: linear warmup then cosine decay, stepped once per batch (same math as the example)
def cosine_warmup(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    total_steps = max(1, total_epochs * steps_per_epoch)
    warmup_steps = min(int(round(warmup_epochs * steps_per_epoch)), total_steps - 1)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# "0003_cand1__blur.jpg" -> "0003", the stem of the original input image
def original_stem(name):
    return name.split("__")[0].split("_cand")[0]


# read labels.csv into {stem: label}
def load_labels(path):
    with open(path) as f:
        return {Path(r["filename"]).stem: int(r["label"]) for r in csv.DictReader(f)}


# turn an already-open PIL image into a normalized (3, n, n) float tensor
def image_to_tensor(im, n):
    im = im.convert("RGB").resize((n, n))
    arr = (np.asarray(im, dtype="float32") / 255.0 - MEAN) / STD
    return torch.from_numpy(arr).permute(2, 0, 1)


# same but starting from a file path
def to_tensor(path, n):
    return image_to_tensor(Image.open(path), n)


# dataset over (image path, label) pairs
class ImageDataset(Dataset):
    def __init__(self, samples, n):
        self.samples = samples
        self.n = n

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return to_tensor(path, self.n), label


# collect (path, label) pairs from the crops or views folder, skipping unlabeled (-1) images
def gather_samples(cfg):
    c = cfg["cnn"]
    folder = Path(cfg["paths"]["views" if c["train_on"] == "views" else "crops"])
    labels = load_labels(c["labels"])
    samples = []
    # detector saves .png (transparent padding), views save .jpg — take both
    for p in sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png"))):
        label = labels.get(original_stem(p.stem), -1)
        if label != -1:
            samples.append((p, label))
    if not samples:
        raise SystemExit(f"no labeled training images found in {folder}")
    return samples


# split samples into train/val so all versions of one input image stay on the same side
def split_train_val(samples, val_split, seed):
    stems = sorted({original_stem(p.stem) for p, _ in samples})
    random.Random(seed).shuffle(stems)
    val_stems = set(stems[: int(len(stems) * val_split)])
    train_s = [s for s in samples if original_stem(s[0].stem) not in val_stems]
    val_s = [s for s in samples if original_stem(s[0].stem) in val_stems]
    return train_s, val_s


# one training epoch, returns the mean batch loss
def run_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total = 0.0
    for xs, ys in loader:
        xs, ys = xs.to(device), ys.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xs), ys)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / max(1, len(loader))


# macro f1 of the model on a loader
@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    y_true, y_pred = [], []
    for xs, ys in loader:
        y_pred += model(xs.to(device)).argmax(1).cpu().tolist()
        y_true += ys.tolist()
    return macro_f1(y_true, y_pred, num_classes)


# train one model: phase 1 with the base frozen (head only), phase 2 with everything trainable
def train_model(mcfg, cfg, device, epoch_hook=None):
    c = cfg["cnn"]
    num_classes = cfg["num_classes"]
    n = cfg["image_size"]
    torch.manual_seed(cfg["seed"])

    train_s, val_s = split_train_val(gather_samples(cfg), c["val_split"], cfg["seed"])
    train_loader = DataLoader(ImageDataset(train_s, n), batch_size=c["batch_size"], shuffle=True)
    val_loader = DataLoader(ImageDataset(val_s, n), batch_size=c["batch_size"])

    weight = class_weights([y for _, y in train_s], num_classes).to(device) if c["weighted_loss"] else None
    criterion = nn.CrossEntropyLoss(label_smoothing=c["label_smoothing"], weight=weight)

    model = build_model(mcfg, num_classes).to(device)
    best_f1, best_state = -1.0, None

    # each phase = (name, freeze the base?, epochs, lr, weight decay)
    phases = [("head", True, mcfg["head"]["epochs"], mcfg["head"]["lr"], 0.0),
              ("full", False, mcfg["full"]["epochs"], mcfg["full"]["lr"], c["weight_decay"])]
    for phase, freeze, epochs, lr, wd in phases:
        if epochs == 0:
            continue
        model.freeze_base(freeze)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        scheduler = cosine_warmup(optimizer, c["warmup_epochs"], epochs, len(train_loader))
        for e in range(epochs):
            if epoch_hook:
                epoch_hook()  # main.py uses this to remake the random views each epoch
            loss = run_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            # keep the best epoch by val f1 (with no val set, keep the newest weights)
            f1 = evaluate(model, val_loader, device, num_classes) if val_s else -1.0
            if f1 >= best_f1:
                best_f1, best_state = f1, copy.deepcopy(model.state_dict())
            print(f"[{mcfg['name']} {phase} {e + 1}/{epochs}] loss={loss:.4f} "
                  f"val_f1={f'{f1:.4f}' if val_s else 'n/a'}")

    model.load_state_dict(best_state)
    dst = Path(c["checkpoints"])
    dst.mkdir(parents=True, exist_ok=True)
    torch.save({"model": mcfg, "state_dict": model.state_dict()}, dst / f"{mcfg['name']}.pt")
    print(f"saved {dst / (mcfg['name'] + '.pt')}")


# train every model listed in the config
def train(cfg, epoch_hook=None):
    device = get_device(cfg)
    for mcfg in cfg["cnn"]["models"]:
        train_model(mcfg, cfg, device, epoch_hook)


# rebuild a model from its checkpoint file
def load_model(mcfg, cfg, device):
    path = Path(cfg["cnn"]["checkpoints"]) / f"{mcfg['name']}.pt"
    if not path.exists():
        raise SystemExit(f"missing {path} — run: uv run python src/cnn.py --train")
    ckpt = torch.load(path, map_location=device)
    # weights come from the checkpoint, so never re-download imagenet weights here
    model = build_model(dict(ckpt["model"], pretrained=False), cfg["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


# score every crop: each model runs the K views in one batched forward, softmax (same as the example)
@torch.no_grad()
def run(cfg):
    c = cfg["cnn"]
    src = Path(cfg["paths"]["views"])
    dst = Path(cfg["paths"]["scores"])
    dst.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg)
    n = cfg["image_size"]
    active = cfg["views"]["active"]
    models = [load_model(m, cfg, device) for m in c["models"]]
    stems = sorted({p.name.split("__")[0] for p in src.glob("*.jpg")})
    for s in stems:
        x = torch.stack([to_tensor(src / f"{s}__{v}.jpg", n) for v in active]).to(device)
        scores = [torch.softmax(m(x), dim=1).cpu().numpy().astype("float32") for m in models]
        # stacked -> (M*K, C) so meta.py works unchanged, 3d -> (M, K, C)
        np.save(dst / f"{s}.npy", np.concatenate(scores) if c["score_layout"] == "stacked" else np.stack(scores))


# ---------------------------------------------------------------------------
# CLI  (python src/cnn.py [--train] [--scratch])
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train the models instead of scoring")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml"))
    train(cfg) if args.train else run(cfg)
