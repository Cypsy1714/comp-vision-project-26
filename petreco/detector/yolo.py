"""Concrete YOLO-backed detector plus a config factory and a verification CLI.

Uses a pretrained, frozen ultralytics YOLO11 (COCO-trained) purely as a
"find the animal" tool. No training happens here. YOLO runs exactly once per
image; the ``max_candidates`` cap bounds downstream cost.

CLI (visual sanity check)::

    python -m petreco.detector.yolo --input <folder> --out <folder>

For each image it saves every candidate crop as ``<stem>_candK.jpg`` (K=0 is the
largest) and prints a one-line summary.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from PIL import Image

from .base import (
    Candidate,
    Detector,
    DetectionResult,
    center_crop_fallback,
    clamp_box,
    crop_with_pad,
    new_signals,
)

# COCO ids kept only as a sanity-check default; the real ids are always resolved
# from ``model.names`` at load time so this survives a differently-ordered model.
_COCO_CAT_ID = 15
_COCO_DOG_ID = 16

# Non-cat/dog animals that should raise the ``other_animal_detected`` signal
# (and optionally become candidates in "all animals" mode).
OTHER_ANIMAL_NAMES = (
    "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
)


def _cfg_get(cfg, key, default):
    """Read ``key`` from a dict-like or attribute-style config, else ``default``."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _resolve_device(device: str) -> str:
    """Map ``"auto"`` to mps-if-available-else-cpu; otherwise honor the string."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class YoloDetector(Detector):
    """Finds every cat/dog candidate with a frozen YOLO11 model.

    Config keys (all optional, sensible defaults shown):
        weights ("yolo11n.pt"), det_conf (0.25), crop_pad (0.08), imgsz (640),
        device ("auto"), target_classes (["cat", "dog"]),
        include_other_animals (False), max_candidates (5),
        on_no_detection ("fallback"). Also accepts ``conf`` as an alias for
        ``det_conf`` and optional ``cat_id`` / ``dog_id`` overrides.
    """

    def __init__(self, cfg=None):
        from ultralytics import YOLO

        self.weights = _cfg_get(cfg, "weights", "yolo11n.pt")
        # ``det_conf`` is the documented key; fall back to legacy ``conf``.
        self.det_conf = float(_cfg_get(cfg, "det_conf", _cfg_get(cfg, "conf", 0.25)))
        self.crop_pad = float(_cfg_get(cfg, "crop_pad", 0.08))
        self.imgsz = int(_cfg_get(cfg, "imgsz", 640))
        self.max_candidates = int(_cfg_get(cfg, "max_candidates", 5))
        self.include_other_animals = bool(_cfg_get(cfg, "include_other_animals", False))
        self.on_no_detection = _cfg_get(cfg, "on_no_detection", "fallback")
        self.device = _resolve_device(_cfg_get(cfg, "device", "auto"))

        self.model = YOLO(self.weights)
        self.names = dict(self.model.names)  # id -> name
        name_to_id = {v: k for k, v in self.names.items()}

        # Sanity-check any configured cat/dog ids against the model's own names.
        for animal, cfg_id, coco_id in (
            ("cat", _cfg_get(cfg, "cat_id", None), _COCO_CAT_ID),
            ("dog", _cfg_get(cfg, "dog_id", None), _COCO_DOG_ID),
        ):
            probe = cfg_id if cfg_id is not None else coco_id
            if self.names.get(probe) != animal and animal in name_to_id:
                warnings.warn(
                    f"class id {probe} is '{self.names.get(probe)}', not '{animal}'; "
                    f"using id {name_to_id[animal]} from model.names instead"
                )

        target_names = list(_cfg_get(cfg, "target_classes", ["cat", "dog"]))
        self.target_class_ids: set[int] = set()
        for name in target_names:
            if name in name_to_id:
                self.target_class_ids.add(name_to_id[name])
            else:
                warnings.warn(f"target class '{name}' not in model.names; skipping")

        self.other_animal_ids: set[int] = {
            name_to_id[n] for n in OTHER_ANIMAL_NAMES if n in name_to_id
        }

        # "All animals" mode: other animals also count as candidates.
        if self.include_other_animals:
            self.target_class_ids |= self.other_animal_ids

    def detect(self, image: Image.Image) -> DetectionResult:
        """Run YOLO once and return all cat/dog candidates + image-level signals."""
        image = image.convert("RGB")
        w, h = image.size
        area = float(w * h)
        signals = new_signals()
        fallback = center_crop_fallback(image)

        results = self.model(
            image,
            conf=self.det_conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes

        # No boxes at all -> pure fallback.
        if boxes is None or len(boxes) == 0:
            signals["was_fallback"] = 1.0
            return DetectionResult([], signals, fallback)

        candidates: list[Candidate] = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].tolist()
            if cls_id in self.target_class_ids:
                box = clamp_box(tuple(xyxy), w, h)
                crop = crop_with_pad(image, box, self.crop_pad)
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                candidates.append(
                    Candidate(
                        crop=crop,
                        box=box,
                        area_fraction=box_area / area if area else 0.0,
                        conf=conf,
                        class_name=self.names.get(cls_id, str(cls_id)),
                    )
                )
            elif cls_id in self.other_animal_ids:
                signals["other_animal_detected"] = 1.0

        # Saw boxes, but none were targets -> fallback (keep other-animal flag).
        if not candidates:
            signals["was_fallback"] = 1.0
            return DetectionResult([], signals, fallback)

        candidates.sort(key=lambda c: c.area_fraction, reverse=True)
        signals["num_catdog_boxes"] = float(len(candidates))  # pre-cap
        signals["max_catdog_conf"] = max(c.conf for c in candidates)
        signals["largest_box_area_fraction"] = candidates[0].area_fraction
        signals["was_fallback"] = 0.0

        return DetectionResult(candidates[: self.max_candidates], signals, fallback)


def build_detector(cfg) -> Detector:
    """Construct a detector from a full pipeline config (reads ``cfg.detector``)."""
    if isinstance(cfg, dict):
        section = cfg.get("detector", {})
    else:
        section = getattr(cfg, "detector", cfg)
    return YoloDetector(section)


# --------------------------------------------------------------------------- #
# CLI: eyeball that the detector finds all cats/dogs.
# --------------------------------------------------------------------------- #
_EXTS = ("*.jpg", "*.jpeg", "*.png")


def _iter_images(folder: Path):
    yield from sorted(p for ext in _EXTS for p in folder.glob(ext))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Visual verification for YoloDetector.")
    parser.add_argument("--input", type=Path, required=True, help="folder of images")
    parser.add_argument("--out", type=Path, required=True, help="folder for candidate crops")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--include-other-animals", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    detector = YoloDetector(
        {
            "weights": args.weights,
            "det_conf": args.conf,
            "imgsz": args.imgsz,
            "device": args.device,
            "max_candidates": args.max_candidates,
            "include_other_animals": args.include_other_animals,
        }
    )

    for path in _iter_images(args.input):
        image = Image.open(path).convert("RGB")
        result = detector.detect(image)
        for k, cand in enumerate(result.candidates):
            cand.crop.save(args.out / f"{path.stem}_cand{k}.jpg")

        s = result.signals
        per_cand = " ".join(
            f"[{k}:{c.class_name} conf={c.conf:.2f} area={c.area_fraction:.3f}]"
            for k, c in enumerate(result.candidates)
        )
        print(
            f"{path.name}: n={len(result.candidates)} "
            f"other_animal={s['other_animal_detected']:.0f} "
            f"fallback={s['was_fallback']:.0f} {per_cand}"
        )


if __name__ == "__main__":
    _main()
