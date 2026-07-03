# throwaway smoke test for cnn.py: fake crops + 2 views from raw images, tiny training, shape checks
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

sys.path.insert(0, "src")
import cnn

root = Path("tmp_cnn_test")
crops, views, scores, ckpts = root / "crops", root / "views", root / "scores", root / "models"
for d in (crops, views, scores):
    d.mkdir(parents=True, exist_ok=True)

for p in sorted(Path("input/images").glob("*.jpg"))[:24]:
    im = Image.open(p).convert("RGB").resize((224, 224))
    im.save(crops / f"{p.stem}.jpg")
    ImageOps.mirror(im).save(views / f"{p.stem}__hflip.jpg")
    im.convert("L").convert("RGB").save(views / f"{p.stem}__gray.jpg")

cfg = yaml.safe_load(open("config.yaml"))
cfg["paths"]["crops"] = str(crops)
cfg["paths"]["views"] = str(views)
cfg["paths"]["scores"] = str(scores)
cfg["views"]["active"] = ["hflip", "gray"]
cfg["cnn"]["checkpoints"] = str(ckpts)
for m in cfg["cnn"]["models"]:
    m["head"]["epochs"] = min(m["head"]["epochs"], 1)
    m["full"]["epochs"] = 1

cnn.train(cfg)

cnn.run(cfg)
arr = np.load(next(iter(sorted(scores.glob("*.npy")))))
print("stacked shape:", arr.shape, "row sums:", arr.sum(1)[:3])
assert arr.shape == (2 * 2, cfg["num_classes"])
assert np.allclose(arr.sum(1), 1.0, atol=1e-4)

cfg["cnn"]["score_layout"] = "3d"
cnn.run(cfg)
arr = np.load(next(iter(sorted(scores.glob("*.npy")))))
print("3d shape:", arr.shape)
assert arr.shape == (2, 2, cfg["num_classes"])
print("ok")
