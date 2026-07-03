"""Detector interface, output contract, and shared cropping helpers.

Stage 1 of the pipeline. A detector's only job is to *find* every cat/dog
candidate in an image and hand them downstream sorted largest-first, together
with a small bundle of image-level signals used later for the reject decision.

A detector never classifies a breed, never picks a winner, and never returns
``-1``. Those decisions belong to later stages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from PIL import Image

# The image-level signals every detector must populate. Names are part of the
# contract consumed downstream by the reject meta-classifier — do not rename.
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
        fallback_crop: A center crop of the image, used by downstream code when
            ``candidates`` is empty.
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


def center_crop_fallback(image: Image.Image, frac: float = 0.9) -> Image.Image:
    """Return a centered crop covering ``frac`` of each dimension.

    Used when the detector finds no target animal, so the classifier still gets
    a reasonable, mostly-centered view instead of the raw frame.
    """
    w, h = image.size
    cw, ch = max(int(w * frac), 1), max(int(h * frac), 1)
    left = (w - cw) // 2
    top = (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch))


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
