"""Stage-1 detector package: interface, dataclasses, and YOLO backend."""

from .base import (
    SIGNAL_KEYS,
    Candidate,
    Detector,
    DetectionResult,
    center_crop_fallback,
    clamp_box,
    crop_with_pad,
    new_signals,
)
from .yolo import YoloDetector, build_detector

__all__ = [
    "SIGNAL_KEYS",
    "Candidate",
    "Detector",
    "DetectionResult",
    "YoloDetector",
    "build_detector",
    "center_crop_fallback",
    "clamp_box",
    "crop_with_pad",
    "new_signals",
]
