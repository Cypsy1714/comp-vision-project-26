from pathlib import Path

from PIL import Image

EXTS = ("*.jpg", "*.jpeg", "*.png")


def run(cfg):
    src = Path(cfg["paths"]["input"])
    dst = Path(cfg["paths"]["crops"])
    dst.mkdir(parents=True, exist_ok=True)
    n = cfg["image_size"]
    for p in sorted(q for e in EXTS for q in src.glob(e)):
        # crop biggest
        im = Image.open(p).convert("RGB").resize((n, n))  # TODO yolo
        im.save(dst / f"{p.stem}.jpg")


if __name__ == "__main__":
    import yaml
    run(yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml")))
