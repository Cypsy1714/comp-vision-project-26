from pathlib import Path
from PIL import ImageEnhance
import random
from PIL import Image, ImageFilter, ImageOps
import numpy as np
from PIL import Image
import math


def make(im, v):
    if v == "hflip":
        return ImageOps.mirror(im)
    if v == "gray":
        return im.convert("L").convert("RGB")
    if v == "blur":
        return im.filter(ImageFilter.GaussianBlur(2))
    if v == "more_blur":
        return im.filter(ImageFilter.GaussianBlur(10))
    if v == "most_blur":
        return im.filter(ImageFilter.GaussianBlur(30))
    if v == "sobel":
        return im.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")
    if v == "perspective":
        w, h = im.size
        coeffs = (
            1, 0.2, -0.1 * w,
            0, 1, 0
        )
        return im.transform((w, h), Image.AFFINE, coeffs)
    if v == "extra_perspective":
        w, h = im.size
        coeffs = (
            2, 0.5, -1 * w,
            0, 1, 0
        )
        return im.transform((w, h), Image.AFFINE, coeffs)
    if v == "sudoku":
        return sudoku_shuffle(im, grid=8)
    if v == "solarize":
        return ImageOps.solarize(im, threshold=128)
    if v == "solarize_light":
        return ImageOps.solarize(im, threshold=192)
    if v == "solarize_dark":
        return ImageOps.solarize(im, threshold=64)
    if v == "equalize":
        return ImageOps.equalize(im)
    if v == "posterize":
        return ImageOps.posterize(im, bits=2)
    
    if v == "wave":
        return wave_effect(im, amplitude=30, frequency=3)
    
    

    if v == "color_jitter":
        im = im.convert("RGB")

        brightness = random.uniform(2, 1.4)
        contrast = random.uniform(5, 1.4)
        saturation = random.uniform(0.5, 1.5)

        im = ImageEnhance.Brightness(im).enhance(brightness)
        im = ImageEnhance.Contrast(im).enhance(contrast)
        im = ImageEnhance.Color(im).enhance(saturation)

        return im

    return im  # clean


def wave_effect(im, amplitude=20, frequency=2):
    im = im.convert("RGB")
    arr = np.array(im)
    h, w, c = arr.shape

    out = np.zeros_like(arr)

    for y in range(h):
        shift = int(amplitude * math.sin(2 * math.pi * frequency * y / h))
        out[y] = np.roll(arr[y], shift, axis=0)

    return Image.fromarray(out)

def sudoku_shuffle(im, grid=4):
    im = im.convert("RGB")
    w, h = im.size

    cell_w = w // grid
    cell_h = h // grid

    pieces = []

    for y in range(grid):
        for x in range(grid):
            box = (
                x * cell_w,
                y * cell_h,
                (x + 1) * cell_w if x < grid - 1 else w,
                (y + 1) * cell_h if y < grid - 1 else h
            )
            pieces.append(im.crop(box))

    random.shuffle(pieces)

    out = im.copy()
    i = 0

    for y in range(grid):
        for x in range(grid):
            out.paste(pieces[i], (x * cell_w, y * cell_h))
            i += 1

    return out


def run(cfg):
    src = Path(cfg["paths"]["crops"])
    dst = Path(cfg["paths"]["views"])
    dst.mkdir(parents=True, exist_ok=True)
    active = cfg["views"]["active"]
    for p in sorted(src.glob("*.jpg")):
        im = Image.open(p).convert("RGB")
        for v in active:  # k views
            make(im, v).save(dst / f"{p.stem}__{v}.jpg")


if __name__ == "__main__":
    import yaml
    run(yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml")))
