from pathlib import Path

import numpy as np


def run(cfg):
    src = Path(cfg["paths"]["views"])
    dst = Path(cfg["paths"]["scores"])
    dst.mkdir(parents=True, exist_ok=True)
    k = len(cfg["views"]["active"])
    c = cfg["num_classes"]
    np.random.seed(cfg["seed"])
    stems = sorted({p.name.split("__")[0] for p in src.glob("*.jpg")})
    for s in stems:
        # random scores
        arr = np.random.rand(k, c).astype("float32")  # TODO cnn
        np.save(dst / f"{s}.npy", arr)


if __name__ == "__main__":
    import yaml
    run(yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml")))
