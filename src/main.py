from pathlib import Path

import numpy as np
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


def predict(image):
    # in-memory
    cfg = load_cfg()
    n = cfg["image_size"]
    image.convert("RGB").resize((n, n))  # TODO yolo + views + cnn
    k = len(cfg["views"]["active"])
    np.random.seed(cfg["seed"])
    prob = np.random.rand(k, cfg["num_classes"]).mean(0)
    cls = int(prob.argmax())
    m = cfg["meta"]
    if m["reject_enabled"] and float(prob.max()) < m["threshold"]:
        cls = -1
    return cls


if __name__ == "__main__":
    run()
