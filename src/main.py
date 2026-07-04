from pathlib import Path

import numpy as np
import torch
import yaml

import detector
import views
import cnn
import meta

CFG = Path(__file__).parent.parent / "config.yaml"

STEPS = {"detector": detector.run, "views": views.run, "cnn": cnn.run, "meta": meta.run}


def load_cfg():
    return yaml.safe_load(open(CFG))


def run(cfg=None):
    cfg = cfg or load_cfg()
    for name in cfg["pipeline"]:
        STEPS[name](cfg)


# remake the stochastic views before every epoch so training sees fresh corruption
# (views.regenerate lists which views are random; deterministic ones stay as-is)
def _regen_hook(cfg):
    regen = cfg["views"].get("regenerate", [])
    if not regen or cfg["cnn"]["train_on"] != "views":
        return None
    return lambda: views.run(cfg, only=regen)


def train(cfg=None):
    # everything from raw images to predictions: crop, views, train cnn, score, train meta, predict
    cfg = cfg or load_cfg()
    # training keeps every image: no-detection images get a full-image fallback crop
    detector.run(cfg, training=True)
    views.run(cfg)
    cnn.train(cfg, epoch_hook=_regen_hook(cfg))
    cnn.run(cfg)
    meta.train(cfg)
    meta.run(cfg)


# yolo + cnn + meta weights, loaded once on the first predict() call
_CACHE = {}


def _load_parts(cfg):
    if not _CACHE:
        device = cnn.get_device(cfg)
        _CACHE["device"] = device
        _CACHE["det"] = detector.build_detector(cfg)
        _CACHE["models"] = [cnn.load_model(m, cfg, device) for m in cfg["cnn"]["models"]]
        _CACHE["meta"] = None
        meta_path = Path("weights/meta.pt")
        if meta_path.exists():
            ckpt = torch.load(meta_path, map_location="cpu")
            s = ckpt["schema"]
            mm = meta.MetaMLP(s["feature_dim"], s["hidden"], s["num_classes"])
            mm.load_state_dict(ckpt["state_dict"])
            _CACHE["meta"] = (mm.eval(), s["feature_dim"])
    return _CACHE


# one crop -> (M, K, C) softmax scores, same as the cnn stage but without files
@torch.no_grad()
def _score_crop(crop, cfg, parts):
    n = cfg["image_size"]
    im = crop.convert("RGB")
    x = torch.stack(
        [cnn.image_to_tensor(views.make(im, v), n) for v in cfg["views"]["active"]]
    ).to(parts["device"])
    scores = [torch.softmax(m(x), dim=1).cpu().numpy().astype("float32") for m in parts["models"]]
    return np.stack(scores)


def predict(image, cfg=None):
    # in-memory version of the folder pipeline: detector -> views -> cnn -> meta
    cfg = cfg or load_cfg()
    parts = _load_parts(cfg)
    m_cfg = cfg["meta"]

    result = parts["det"].detect(image)
    if not result.candidates and cfg["detector"].get("reject_on_no_animal", False):
        return -1
    crops = [c.crop for c in result.candidates] or [result.fallback_crop]

    # score every candidate, keep the strongest peak (same dedup rule as meta.py)
    arr3d = max((_score_crop(c, cfg, parts) for c in crops), key=lambda a: float(a.max()))

    if parts["meta"] is not None:
        mm, dim = parts["meta"]
        feat = meta.build_features(arr3d)
        if feat.shape[0] == dim:
            with torch.no_grad():
                cls_logits, rej_logit = mm(torch.tensor(feat).unsqueeze(0))
            if m_cfg["reject_enabled"] and torch.sigmoid(rej_logit).item() > m_cfg["threshold"]:
                return -1
            return int(cls_logits.argmax(dim=1).item())

    # no meta weights yet: mean over models and views + threshold reject
    prob = arr3d.mean(axis=(0, 1))
    cls = int(prob.argmax())
    if m_cfg["reject_enabled"] and float(prob.max()) < m_cfg["threshold"]:
        cls = -1
    return cls


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                        help="train cnn + meta first, then run the whole pipeline")
    args = parser.parse_args()
    train() if args.train else run()
