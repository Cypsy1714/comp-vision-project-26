import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# score-file normalization  →  (M, K, C)
# ---------------------------------------------------------------------------

def normalize_scores(arr: np.ndarray, K: int, C: int) -> np.ndarray:
    """
    Accept any of the three layouts Cem's cnn.py can produce and return (M, K, C).

    Layout rules
    ────────────
    3D  (M, K, C)  → use as-is (validate inner dims)
    2D  (K, C)     → single model; M = 1
    2D  (M*K, C)   → stacked layout; rows must be divisible by K
    Anything else  → ValueError with the exact shape and expected K, C.
    """
    arr = arr.astype("float32")

    if arr.ndim == 3:
        if arr.shape[1] != K or arr.shape[2] != C:
            raise ValueError(
                f"3D score array shape {arr.shape} does not match expected "
                f"(M, K={K}, C={C})"
            )
        return arr

    if arr.ndim == 2:
        rows, cols = arr.shape
        if cols != C:
            raise ValueError(
                f"Score array has {cols} columns but num_classes (C) = {C}."
            )
        if rows == K:
            return arr.reshape(1, K, C)          # single model
        if rows % K == 0:
            return arr.reshape(rows // K, K, C)  # stacked → 3D
        raise ValueError(
            f"Score array shape {arr.shape}: rows={rows} is not K={K} and not "
            f"divisible by K={K} — cannot determine M. "
            f"Check cnn.score_layout in config.yaml."
        )

    raise ValueError(
        f"Score array has ndim={arr.ndim}, shape={arr.shape} — "
        f"expected 2D or 3D."
    )


# ---------------------------------------------------------------------------
# feature building
# ---------------------------------------------------------------------------

def _per_view_entropy(s: np.ndarray) -> np.ndarray:
    """Shannon entropy of each view's softmax distribution. (K, C) → (K,)."""
    p = np.clip(s, 1e-9, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def _top2_margin(s: np.ndarray) -> np.ndarray:
    """top-1 minus top-2 probability for each view. (K, C) → (K,)."""
    desc = np.sort(s, axis=1)[:, ::-1]   # (K, C) sorted descending
    return desc[:, 0] - desc[:, 1]


def build_features(arr3d: np.ndarray) -> np.ndarray:
    """
    Build the meta-feature vector from a (M, K, C) score array.

    Per-model block (length = K*C + 2C + 2K + 1  per model):
      flat softmax         K*C   — raw probability landscape
      cross-view mean      C     — average class confidence across views
      cross-view variance  C     — stability of confidence across views
      per-view entropy     K     — how uncertain each view is
      top-1 agreement      1     — fraction of views that agree on the top class
      per-view top-2 margin K    — gap between 1st and 2nd class per view

    Cross-model block (only when M > 1, length = 2):
      mean-abs-diff        1     — avg absolute difference between models' mean vectors
      argmax agreement     1     — 1 if all models point to the same class, 0 otherwise

    Total dimension = M * (K*C + 2C + 2K + 1) + (2 if M > 1 else 0)
    This is stored in the checkpoint schema and checked at load time.
    """
    M, K, C = arr3d.shape
    blocks = []
    per_model_means = []

    for m in range(M):
        s = arr3d[m]                                 # (K, C)
        mean_v = s.mean(axis=0)                      # (C,)
        block = np.concatenate([
            s.flatten(),                             # K*C
            mean_v,                                  # C
            s.var(axis=0),                           # C
            _per_view_entropy(s),                    # K
            [float((s.argmax(axis=1) == s.argmax(axis=1)[0]).mean())],  # 1
            _top2_margin(s),                         # K
        ])
        blocks.append(block)
        per_model_means.append(mean_v)

    feat = np.concatenate(blocks)

    if M > 1:
        means = np.stack(per_model_means)            # (M, C)
        # mean of pairwise absolute differences between models' mean softmax vectors
        diffs = [
            float(np.abs(means[i] - means[j]).mean())
            for i in range(M) for j in range(i + 1, M)
        ]
        cross_diff = float(np.mean(diffs))
        # 1 if all models pick the same overall top class, else 0
        tops = [int(means[m].argmax()) for m in range(M)]
        argmax_agree = float(len(set(tops)) == 1)
        feat = np.concatenate([feat, [cross_diff, argmax_agree]])

    return feat.astype("float32")


def feature_dim(M: int, K: int, C: int) -> int:
    """Expected feature vector length for M models, K views, C classes."""
    return M * (K * C + 2 * C + 2 * K + 1) + (2 if M > 1 else 0)


# ---------------------------------------------------------------------------
# score-file loading + candidate deduplication
# ---------------------------------------------------------------------------

def _load_scores(scores_dir: Path, K: int, C: int):
    """
    Load every .npy in scores_dir, normalize to (M, K, C), deduplicate _cand stems.

    Filename mapping:  "0003_cand1" → original stem "0003"
    Dedup rule: if multiple candidates map to the same original stem, keep the
    one whose arr3d.max() is highest (strongest peak confidence across all
    models and views — indicates the best-cropped detection).

    Returns
    ───────
    stem_feats  : {original_stem: feature_vector (float32 ndarray)}
    stem_arrays : {original_stem: (M, K, C) array}
    """
    # best_conf tracks (confidence, arr3d) per original stem
    best: dict[str, tuple[float, np.ndarray]] = {}

    for p in sorted(scores_dir.glob("*.npy")):
        arr = np.load(p)
        arr3d = normalize_scores(arr, K, C)
        orig = p.stem.split("_cand")[0]
        conf = float(arr3d.max())
        if orig not in best or conf > best[orig][0]:
            best[orig] = (conf, arr3d)

    stem_arrays = {s: v[1] for s, v in best.items()}
    stem_feats  = {s: build_features(a) for s, a in stem_arrays.items()}
    return stem_feats, stem_arrays


# ---------------------------------------------------------------------------
# Cem's val-split — replicated exactly from origin/cem:src/cnn.py
# ---------------------------------------------------------------------------

def _cem_val_stems(cfg) -> set[str]:
    """
    Reproduce split_train_val from Cem's cnn.py so we can identify which stems
    his CNN never trained on (safe to use as meta-training data without leakage).

    From Cem's code:
        stems = sorted({original_stem(p.stem) for p, _ in samples})
        random.Random(seed).shuffle(stems)
        val_stems = set(stems[: int(len(stems) * val_split)])

    "samples" in his code is every labeled (non-(-1)) crop/view file.
    We reconstruct that stem set from labels.csv.
    """
    labels_path = Path(cfg["cnn"].get("labels", "input/images/labels.csv"))
    val_split   = cfg["cnn"].get("val_split", 0.2)
    seed        = cfg["seed"]

    with open(labels_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # only the non-confounder images, matching Cem's gather_samples filter
    stems = sorted({Path(r["filename"]).stem for r in rows if int(r["label"]) != -1})
    random.Random(seed).shuffle(stems)
    n_val = int(len(stems) * val_split)
    return set(stems[:n_val])


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MetaMLP(nn.Module):
    """
    Shared trunk (hidden layers) that branches into two heads:
      class_head  — num_classes logits  (CrossEntropyLoss for breed prediction)
      reject_head — 1 logit             (BCEWithLogitsLoss; sigmoid > threshold → -1)
    """

    def __init__(self, input_dim: int, hidden: list, num_classes: int):
        super().__init__()
        layers, in_dim = [], input_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.trunk       = nn.Sequential(*layers)
        self.class_head  = nn.Linear(in_dim, num_classes)
        self.reject_head = nn.Linear(in_dim, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.class_head(h), self.reject_head(h)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train(cfg, oof: bool = False):
    if oof:
        print("[meta] --oof: k-fold OOF path is reserved for a future iteration. "
              "Using Cem's val-split strategy instead.")

    scores_dir = Path(cfg["paths"]["scores"])
    K     = len(cfg["views"]["active"])
    C     = cfg["num_classes"]
    m_cfg = cfg["meta"]
    torch.manual_seed(cfg["seed"])

    # load and normalize all score files
    stem_feats, stem_arrays = _load_scores(scores_dir, K, C)
    if not stem_feats:
        raise SystemExit(f"[meta] No .npy files found in {scores_dir}. Run cnn stage first.")

    # ground-truth labels
    labels_path = Path(cfg["cnn"].get("labels", "input/images/labels.csv"))
    with open(labels_path, newline="") as f:
        gt = {Path(r["filename"]).stem: int(r["label"]) for r in csv.DictReader(f)}

    # which stems went into Cem's val set (he never trained on them → no leakage)
    val_stems        = _cem_val_stems(cfg)
    confounder_stems = {s for s, lbl in gt.items() if lbl == -1}

    # meta training set: Cem's val-split breeds + all confounders that have a score file
    train_stems = (val_stems | confounder_stems) & set(stem_feats)

    print(f"\n[meta] training stems: {len(train_stems)} total")
    print(f"  val-split breeds available in scores: "
          f"{sorted(val_stems & set(stem_feats))}")
    print(f"  confounder stems available in scores: "
          f"{sorted(confounder_stems & set(stem_feats))}")

    if not train_stems:
        raise SystemExit(
            "[meta] No usable training stems (need val-split breed or confounder score files)."
        )

    # build training tensors
    feats_list, class_labels, reject_labels = [], [], []
    for stem in sorted(train_stems):
        true_lbl = gt.get(stem, -1)
        arr3d    = stem_arrays[stem]
        feat     = stem_feats[stem]

        # CNN top-1: mean over all M models and K views, then argmax
        cnn_top1 = int(arr3d.mean(axis=(0, 1)).argmax())

        # reject = 1 if confounder OR CNN got it wrong (task specification)
        reject = 1 if (true_lbl == -1 or cnn_top1 != true_lbl) else 0

        feats_list.append(feat)
        class_labels.append(true_lbl)
        reject_labels.append(reject)

    feat_dim_val = feats_list[0].shape[0]
    X     = torch.tensor(np.stack(feats_list), dtype=torch.float32)
    y_cls = torch.tensor(class_labels,         dtype=torch.long)
    y_rej = torch.tensor(reject_labels,         dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, y_cls, y_rej)
    loader  = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)

    model     = MetaMLP(feat_dim_val, m_cfg["hidden"], C)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce_loss   = nn.CrossEntropyLoss()
    bce_loss  = nn.BCEWithLogitsLoss()

    print(f"\n[meta] MLP  input={feat_dim_val}  hidden={m_cfg['hidden']}  classes={C}")
    print("[meta] training for 100 epochs ...")

    for epoch in range(100):
        model.train()
        ep_loss = 0.0
        for xb, yb_cls, yb_rej in loader:
            optimizer.zero_grad()
            cls_logits, rej_logit = model(xb)

            loss_rej = bce_loss(rej_logit, yb_rej)

            # class loss only on rows that are actual breeds (not confounders)
            breed_mask = yb_cls != -1
            loss_cls = (
                ce_loss(cls_logits[breed_mask], yb_cls[breed_mask])
                if breed_mask.any()
                else torch.tensor(0.0)
            )

            loss = loss_rej + loss_cls
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}/100  loss={ep_loss / len(loader):.4f}")

    schema = {
        "feature_dim": feat_dim_val,
        "num_classes": C,
        "hidden":      m_cfg["hidden"],
    }
    Path("weights").mkdir(exist_ok=True)
    torch.save({"schema": schema, "state_dict": model.state_dict()}, "weights/meta.pt")
    print(f"\n[meta] saved weights/meta.pt  schema={schema}")


# ---------------------------------------------------------------------------
# fallback: mean over all rows + threshold  (mirrors the original stub)
# ---------------------------------------------------------------------------

def _run_fallback(cfg):
    """
    If weights/meta.pt is missing or incompatible, fall back to the
    original stub behaviour: mean over ALL score rows + threshold reject.
    Handles _cand dedup by keeping the candidate with the highest max prob.
    """
    scores_dir = Path(cfg["paths"]["scores"])
    dst        = Path(cfg["paths"]["preds"])
    dst.mkdir(parents=True, exist_ok=True)
    m_cfg = cfg["meta"]

    # best candidate per original stem (track by max prob over all rows)
    best: dict[str, tuple[float, int]] = {}   # orig_stem → (confidence, label)
    for p in sorted(scores_dir.glob("*.npy")):
        arr  = np.load(p)
        prob = arr.mean(0)
        cls  = int(prob.argmax())
        if m_cfg["reject_enabled"] and float(prob.max()) < m_cfg["threshold"]:
            cls = -1
        conf = float(arr.max())
        orig = p.stem.split("_cand")[0]
        if orig not in best or conf > best[orig][0]:
            best[orig] = (conf, cls)

    with open(dst / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        for stem in sorted(best):
            w.writerow([f"{stem}.jpg", best[stem][1]])


# ---------------------------------------------------------------------------
# public entry point — called by main.py
# ---------------------------------------------------------------------------

def run(cfg):
    scores_dir = Path(cfg["paths"]["scores"])
    dst        = Path(cfg["paths"]["preds"])
    dst.mkdir(parents=True, exist_ok=True)
    K     = len(cfg["views"]["active"])
    C     = cfg["num_classes"]
    m_cfg = cfg["meta"]

    weights_path = Path("weights/meta.pt")

    # ── guard: missing weights ─────────────────────────────────────────────
    if not weights_path.exists():
        print("[meta] WARNING: weights/meta.pt not found — "
              "falling back to mean+threshold (run: python src/meta.py --train)")
        _run_fallback(cfg)
        return

    # ── load score files ───────────────────────────────────────────────────
    try:
        stem_feats, _ = _load_scores(scores_dir, K, C)
    except Exception as exc:
        print(f"[meta] WARNING: score loading failed ({exc}) — "
              "falling back to mean+threshold")
        _run_fallback(cfg)
        return

    if not stem_feats:
        print("[meta] WARNING: no .npy files found — falling back to mean+threshold")
        _run_fallback(cfg)
        return

    # ── guard: feature-dim mismatch ────────────────────────────────────────
    ckpt   = torch.load(weights_path, map_location="cpu")
    schema = ckpt["schema"]

    actual_dim = next(iter(stem_feats.values())).shape[0]
    stored_dim = schema["feature_dim"]
    if actual_dim != stored_dim:
        print(
            f"[meta] WARNING: feature dim mismatch — "
            f"current scores → {actual_dim} dims, checkpoint expects {stored_dim} dims. "
            f"Retrain with: python src/meta.py --train. Falling back to mean+threshold."
        )
        _run_fallback(cfg)
        return

    # ── MLP inference ──────────────────────────────────────────────────────
    model = MetaMLP(stored_dim, schema["hidden"], schema["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    threshold      = m_cfg["threshold"]
    reject_enabled = m_cfg["reject_enabled"]
    rows = []

    with torch.no_grad():
        for stem in sorted(stem_feats):
            x = torch.tensor(stem_feats[stem]).unsqueeze(0)
            cls_logits, rej_logit = model(x)
            if reject_enabled and torch.sigmoid(rej_logit).item() > threshold:
                label = -1
            else:
                label = int(cls_logits.argmax(dim=1).item())
            rows.append((f"{stem}.jpg", label))

    with open(dst / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        w.writerows(rows)

    print(f"[meta] predictions → {dst / 'predictions.csv'}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml"))

    parser = argparse.ArgumentParser(description="Stage D — meta classifier")
    parser.add_argument("--train", action="store_true",
                        help="train MLP and save weights/meta.pt")
    parser.add_argument("--oof",   action="store_true",
                        help="(reserved) use k-fold OOF features instead of val-split")
    args = parser.parse_args()

    if args.train:
        train(cfg, oof=args.oof)
    else:
        run(cfg)
