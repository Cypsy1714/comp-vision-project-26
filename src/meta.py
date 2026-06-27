import csv
from pathlib import Path

import numpy as np


def run(cfg):
    src = Path(cfg["paths"]["scores"])
    dst = Path(cfg["paths"]["preds"])
    dst.mkdir(parents=True, exist_ok=True)
    m = cfg["meta"]
    rows = []
    for p in sorted(src.glob("*.npy")):
        prob = np.load(p).mean(0)  # avg views
        cls = int(prob.argmax())
        if m["reject_enabled"] and float(prob.max()) < m["threshold"]:
            cls = -1  # reject
        rows.append((f"{p.stem}.jpg", cls))
    with open(dst / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        w.writerows(rows)


if __name__ == "__main__":
    import yaml
    run(yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml")))
