from __future__ import annotations
from pathlib import Path
import warnings
from pathlib import Path
from PIL import Image
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch
from ultralytics import YOLO
import yaml


### Base Functions


SIGNAL_KEYS = (
    "num_catdog_boxes",          # count of cat/dog detections (pre-cap)
    "max_catdog_conf",           # max confidence among them, 0.0 if none
    "largest_box_area_fraction", # largest candidate box area / image area, 0.0 if none
    "other_animal_detected",     # 1.0 if any non-cat/dog animal seen, else 0.0
    "was_fallback",              # 1.0 if no cat/dog found, else 0.0
)

def new_signals() -> dict[str, float]:
    """Return a fresh signals dict with every key initialised to ``0.0``."""
    return {k: 0.0 for k in SIGNAL_KEYS}

@dataclass
class Candidate:
    """One detected cat/dog (or other animal in "all animals" mode).

    Attributes:
        crop: The padded RGB crop of this animal, ready for the classifier.
        box: Clamped detection box ``(x1, y1, x2, y2)`` in pixel coordinates.
        area_fraction: Box area divided by the full image area, in ``[0, 1]``.
        conf: Detector confidence for this box.
        class_name: COCO class name, e.g. ``"cat"`` or ``"dog"``.
    """

    crop: Image.Image
    box: tuple[float, float, float, float]
    area_fraction: float
    conf: float
    class_name: str

@dataclass
class DetectionResult:
    """Everything Stage 1 hands to the rest of the pipeline for one image.

    Attributes:
        candidates: Cat/dog crops sorted by area DESC and capped at
            ``max_candidates``. Empty when nothing was found.
        signals: Exactly the five :data:`SIGNAL_KEYS`, all floats.
        fallback_crop: The full image resized to the output square, used by
            downstream code when ``candidates`` is empty.
    """

    candidates: list[Candidate]
    signals: dict[str, float]
    fallback_crop: Image.Image = field(repr=False)

def clamp_box(
    box: tuple[float, float, float, float], w: int, h: int
) -> tuple[float, float, float, float]:
    """Clamp ``(x1, y1, x2, y2)`` to the image and guarantee a non-empty box.

    Coordinates are clipped to ``[0, w] x [0, h]`` and ordered so ``x1 < x2``
    and ``y1 < y2``, widening by one pixel if the box collapsed.
    """
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(max(x1, 0.0), float(w))
    x2 = min(max(x2, 0.0), float(w))
    y1 = min(max(y1, 0.0), float(h))
    y2 = min(max(y2, 0.0), float(h))
    if x2 <= x1:
        x1 = min(x1, float(w - 1))
        x2 = min(x1 + 1.0, float(w))
    if y2 <= y1:
        y1 = min(y1, float(h - 1))
        y2 = min(y1 + 1.0, float(h))
    return (x1, y1, x2, y2)

def crop_with_pad(
    image: Image.Image, box: tuple[float, float, float, float], pad: float
) -> Image.Image:
    """Crop ``image`` to ``box`` after expanding it by ``pad`` on every side.

    ``pad`` is a fraction of the box's own width/height (e.g. ``0.08`` adds 8%
    context on each edge). The expanded box is clamped to the image bounds and
    guaranteed non-empty, so the returned crop always has a positive size.
    """
    w, h = image.size
    x1, y1, x2, y2 = clamp_box(box, w, h)
    dx = (x2 - x1) * pad
    dy = (y2 - y1) * pad
    padded = clamp_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), w, h)
    left, top, right, bottom = (int(round(v)) for v in padded)
    right = max(right, left + 1)
    bottom = max(bottom, top + 1)
    return image.crop((left, top, right, bottom))

# Lanczos resampling anti-aliases whether the crop is scaled up or down.
_RESAMPLE = Image.Resampling.LANCZOS


def resize_to_output(
    image: Image.Image, size: int, mode: str = "fit"
) -> Image.Image:
    """Resize ``image`` to a ``size`` x ``size`` square with anti-aliasing.

    Lanczos resampling is used, which anti-aliases both when zooming in
    (upscaling) and zooming out (downscaling).

    Modes:
        ``"stretch"``: ignore the aspect ratio and distort the crop so it fills
            the whole square. Keeps the input mode (e.g. RGB).
        ``"fit"``: preserve the aspect ratio, scale the crop to fit inside the
            square, and pad the leftover space with fully transparent pixels,
            centering the crop. Returns an RGBA image.
    """
    size = max(int(size), 1)
    if mode == "stretch":
        return image.resize((size, size), _RESAMPLE)
    # "fit": aspect-preserving, transparent letterbox padding.
    w, h = image.size
    scale = min(size / w, size / h)
    nw = max(int(round(w * scale)), 1)
    nh = max(int(round(h * scale)), 1)
    resized = image.resize((nw, nh), _RESAMPLE).convert("RGBA")
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas

class Detector(ABC):
    """Abstract Stage-1 detector.

    Subclasses implement :meth:`detect`. Everything else (the single-crop
    compatibility wrapper, the helpers above) is shared so alternate detection
    backends stay swappable via config.
    """

    @abstractmethod
    def detect(self, image: Image.Image) -> DetectionResult:
        """Find all cat/dog candidates and compute image-level signals."""
        raise NotImplementedError

    def detect_and_crop(self, image: Image.Image) -> tuple[Image.Image, dict[str, float]]:
        """Compat wrapper for the old single-crop training path.

        Returns the largest candidate's crop (or the center-crop fallback when
        there are none) together with the signals dict.
        """
        result = self.detect(image)
        crop = result.candidates[0].crop if result.candidates else result.fallback_crop
        return crop, result.signals


### YOLO Functions


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
        include_other_animals (False), max_candidates (5). Also accepts
        ``conf`` as an alias for ``det_conf`` and optional ``cat_id`` /
        ``dog_id`` overrides.
    """

    def __init__(self, cfg=None):
        self.weights = _cfg_get(cfg, "weights", "yolo11n.pt")
        # ``det_conf`` is the documented key; fall back to legacy ``conf``.
        self.det_conf = float(_cfg_get(cfg, "det_conf", _cfg_get(cfg, "conf", 0.25)))
        self.crop_pad = float(_cfg_get(cfg, "crop_pad", 0.08))
        self.imgsz = int(_cfg_get(cfg, "imgsz", 640))
        self.max_candidates = int(_cfg_get(cfg, "max_candidates", 5))
        self.include_other_animals = bool(_cfg_get(cfg, "include_other_animals", False))
        self.device = _resolve_device(_cfg_get(cfg, "device", "auto"))

        # Every saved crop is forced to this square size. ``output_size`` is the
        # documented detector key; fall back to the shared top-level image_size.
        self.output_size = int(
            _cfg_get(cfg, "output_size", _cfg_get(cfg, "image_size", 224))
        )
        # "fit" = keep aspect ratio + transparent padding; "stretch" = distort.
        self.resize_mode = str(_cfg_get(cfg, "resize_mode", "fit")).lower()
        if self.resize_mode not in ("fit", "stretch"):
            warnings.warn(
                f"unknown resize_mode '{self.resize_mode}'; falling back to 'fit'"
            )
            self.resize_mode = "fit"

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
        fallback = resize_to_output(image, self.output_size, self.resize_mode)

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
                crop = resize_to_output(
                    crop_with_pad(image, box, self.crop_pad),
                    self.output_size,
                    self.resize_mode,
                )
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
        section = dict(cfg.get("detector", {}))
        # Let the detector fall back to the shared top-level image_size when it
        # has no output_size of its own.
        section.setdefault("image_size", cfg.get("image_size", 224))
    else:
        section = getattr(cfg, "detector", cfg)
    return YoloDetector(section)


# --------------------------------------------------------------------------- #
# CLI: eyeball that the detector finds all cats/dogs.
# --------------------------------------------------------------------------- #

EXTS = ("*.jpg", "*.jpeg", "*.png")


def _iter_images(folder: Path):
    yield from sorted(p for ext in EXTS for p in folder.glob(ext))


def run(cfg, training=False):
    src = Path(cfg["paths"]["input"])
    dst = Path(cfg["paths"]["crops"])
    dst.mkdir(parents=True, exist_ok=True)

    detector = build_detector(cfg)
    reject = bool(_cfg_get(cfg.get("detector", {}), "reject_on_no_animal", False))

    for path in _iter_images(src):
        image = Image.open(path).convert("RGB")
        result = detector.detect(image)
        for k, cand in enumerate(result.candidates):
            # Transparent-padded ("fit") crops need PNG to keep the alpha.
            ext = "png" if cand.crop.mode == "RGBA" else "jpg"
            cand.crop.save(dst / f"{path.stem}_cand{k}.{ext}")

        # No animal found: with reject_on_no_animal we write nothing so the
        # image is rejected downstream; otherwise (and always in training, so
        # no image is lost) the full-image fallback becomes the crop.
        if not result.candidates and (training or not reject):
            fb = result.fallback_crop
            ext = "png" if fb.mode == "RGBA" else "jpg"
            fb.save(dst / f"{path.stem}_cand0.{ext}")

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
    run(yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml")))
