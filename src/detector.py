from pathlib import Path

from PIL import Image

EXTS = ("*.jpg", "*.jpeg", "*.png")

# Anti-aliasing filter used for every resize (good for up- and down-scaling).
RESAMPLE = Image.Resampling.LANCZOS


def _fit_size(size, target):
    """Scale (w, h) down/up to fit inside target x target, keeping aspect ratio."""
    w, h = size
    scale = min(target / w, target / h)
    return max(1, round(w * scale)), max(1, round(h * scale))


def resize_output(im, n, mode):
    """Return an n x n image using the chosen mode.

    - "stretch": distort the image to fill the whole n x n frame.
    - "pad":     keep the aspect ratio, center it, and fill the leftover space
                 with transparent pixels (transparent background).
    Both paths use LANCZOS anti-aliasing when scaling.
    """
    if mode == "pad":
        fitted = im.convert("RGBA").resize(_fit_size(im.size, n), RESAMPLE)
        canvas = Image.new("RGBA", (n, n), (0, 0, 0, 0))  # fully transparent
        w, h = fitted.size
        canvas.paste(fitted, ((n - w) // 2, (n - h) // 2))
        return canvas
    return im.resize((n, n), RESAMPLE)


def run(cfg):
    src = Path(cfg["paths"]["input"])
    dst = Path(cfg["paths"]["crops"])
    dst.mkdir(parents=True, exist_ok=True)
    n = cfg["image_size"]
    # config toggle: "stretch" (default) or "pad" (transparent letterbox)
    mode = cfg.get("detector", {}).get("resize_mode", "stretch")
    ext = "png" if mode == "pad" else "jpg"  # png keeps the transparency
    for p in sorted(q for e in EXTS for q in src.glob(e)):
        # crop biggest
        im = Image.open(p).convert("RGB")  # TODO yolo
        out = resize_output(im, n, mode)
        out.save(dst / f"{p.stem}.{ext}")


if __name__ == "__main__":
    import yaml
    run(yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml")))
