from pathlib import Path
from PIL import ImageEnhance
import random
from PIL import Image, ImageFilter, ImageOps, ImageChops, ImageEnhance, ImageDraw
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
    
    if v == "cartoon":
         base = ImageOps.posterize(im.convert("RGB"), 3)
         base = base.filter(ImageFilter.SMOOTH_MORE)

         edges = im.convert("L").filter(ImageFilter.FIND_EDGES)
         edges = ImageOps.invert(edges)
         edges = edges.point(lambda x: 255 if x > 120 else 0)
         edges = edges.convert("RGB")

         return ImageChops.multiply(base, edges)
    
    if v == "comic_clean":
        return ligne_claire(im)
    
    if v == "random_sudoku_region":
        return random_region_sudoku(im, region_size=0.45, grid=4)
    
    if v == "wave":
        return wave_effect(im, amplitude=30, frequency=3)
    

    if v == "pop_art":
         im = im.convert("RGB")

    # Kontrast ve renkleri artır
         im = ImageEnhance.Contrast(im).enhance(1.8)
         im = ImageEnhance.Color(im).enhance(2.2)

    # Renkleri azalt / poster etkisi
         im = ImageOps.posterize(im, bits=3)

         arr = np.array(im)

    # Renkleri daha parlak ve sert hale getir
         arr[:, :, 0] = np.where(arr[:, :, 0] > 128, 255, 40)
         arr[:, :, 1] = np.where(arr[:, :, 1] > 128, 220, 30)
         arr[:, :, 2] = np.where(arr[:, :, 2] > 128, 180, 60)

         return Image.fromarray(arr.astype(np.uint8))
    

    if v == "strong_pop_art":
        im = im.convert("RGB")
        im = ImageOps.posterize(im, bits=2)

        arr = np.array(im)
 
        palettes = [
            [255, 30, 80],
            [30, 220, 255],
            [255, 230, 30],
            [120, 40, 255],
        ]

        gray = np.array(im.convert("L"))

        out = np.zeros_like(arr)

        out[gray < 64] = palettes[3]
        out[(gray >= 64) & (gray < 128)] = palettes[0]
        out[(gray >= 128) & (gray < 192)] = palettes[1]
        out[gray >= 192] = palettes[2]

        return Image.fromarray(out.astype(np.uint8))
    

    if v == "inception_fold":
        return inception_fold(im)

    if v == "warhol_random_fill":
        return warhol_random_fill(im)

    if v == "picasso":
        im = im.convert("RGB")
        w, h = im.size

    # Renkleri sertleştir
        base = ImageEnhance.Color(im).enhance(2.0)
        base = ImageEnhance.Contrast(base).enhance(1.6)
        base = ImageOps.posterize(base, bits=3)

        out = base.copy()

    # Resmi rastgele parçalara bölüp oynat
        for _ in range(88):
            bw = random.randint(w // 10, w // 5)
            bh = random.randint(h // 10, h // 5)

            x = random.randint(0, w - bw)
            y = random.randint(0, h - bh)

            piece = base.crop((x, y, x + bw, y + bh))

        # Bazı parçaları aynala / döndür
            if random.random() < 0.5:
                piece = ImageOps.mirror(piece)

            angle = random.choice([-15, -8, 8, 15])
            piece = piece.rotate(angle, resample=Image.BICUBIC, expand=False)

            nx = min(max(x + random.randint(-w // 10, w // 10), 0), w - bw)
            ny = min(max(y + random.randint(-h // 10, h // 10), 0), h - bh)

            out.paste(piece, (nx, ny))

        return out
    


    if v == "triangle_sudoku":
        return triangle_sudoku(im, grid=2)
    

    if v == "hide_and_seek":
        return hide_and_seek(im, grid=4, hide_prob=0.5)
    
    if v == "random_value_fill":
        return random_value_fill(im)
    

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

def warhol_random_fill(im):
    im = im.convert("RGB")

    # Warhol / pop-art görünümü
    im = ImageEnhance.Contrast(im).enhance(1.8)
    im = ImageEnhance.Color(im).enhance(2.4)
    im = ImageOps.posterize(im, bits=2)

    arr = np.array(im).copy()
    gray = np.array(im.convert("L"))

    out = np.zeros_like(arr)

    out[gray < 64] = [120, 40, 255]
    out[(gray >= 64) & (gray < 128)] = [255, 30, 80]
    out[(gray >= 128) & (gray < 192)] = [30, 220, 255]
    out[gray >= 192] = [255, 230, 30]

    # Random value fill
    h, w, _ = out.shape

    fill_w = random.randint(w // 8, w // 3)
    fill_h = random.randint(h // 8, h // 3)

    x = random.randint(0, w - fill_w)
    y = random.randint(0, h - fill_h)

    random_patch = np.random.randint(
        0, 256,
        size=(fill_h, fill_w, 3),
        dtype=np.uint8
    )

    out[y:y+fill_h, x:x+fill_w] = random_patch

    return Image.fromarray(out.astype(np.uint8))

def random_value_fill(im):
    arr = np.array(im.convert("RGB")).copy()

    h, w, _ = arr.shape

    fill_w = random.randint(w // 8, w // 3)
    fill_h = random.randint(h // 8, h // 3)

    x = random.randint(0, w - fill_w)
    y = random.randint(0, h - fill_h)

    random_patch = np.random.randint(
        0, 256,
        size=(fill_h, fill_w, 3),
        dtype=np.uint8
    )

    arr[y:y+fill_h, x:x+fill_w] = random_patch

    return Image.fromarray(arr)

def hide_and_seek(im, grid=4, hide_prob=0.5):
    im = im.convert("RGB")
    arr = np.array(im).copy()

    h, w, _ = arr.shape
    cell_w = w // grid
    cell_h = h // grid

    for y in range(grid):
        for x in range(grid):
            if random.random() < hide_prob:
                x0 = x * cell_w
                y0 = y * cell_h
                x1 = (x + 1) * cell_w if x < grid - 1 else w
                y1 = (y + 1) * cell_h if y < grid - 1 else h

                arr[y0:y1, x0:x1] = [0, 0, 0]

    return Image.fromarray(arr)

def inception_fold(im):
    im = im.convert("RGB")
    w, h = im.size

    out = im.copy()

    # Alt kısmı zemin gibi al
    ground = im.crop((0, h // 2, w, h))

    # Zemini dikey çevir: yukarı kıvrılmış gibi
    folded = ImageOps.flip(ground)

    # Daha karanlık/kontrastlı yap
    folded = ImageEnhance.Contrast(folded).enhance(1.3)
    folded = ImageEnhance.Brightness(folded).enhance(0.85)

    # Üst yarıya sığdır
    folded = folded.resize((w, h // 2))

    # Hafif transparan bindirme hissi
    out.paste(folded, (0, 0))

    return out

def triangle_sudoku(im, grid=4):
    im = im.convert("RGB")
    w, h = im.size

    cell_w = w // grid
    cell_h = h // grid

    triangles = []

    for y in range(grid):
        for x in range(grid):
            x0 = x * cell_w
            y0 = y * cell_h
            x1 = (x + 1) * cell_w if x < grid - 1 else w
            y1 = (y + 1) * cell_h if y < grid - 1 else h

            cell = im.crop((x0, y0, x1, y1))
            cw, ch = cell.size

            # Üçgen 1 maskesi
            mask1 = Image.new("L", (cw, ch), 0)
            draw1 = ImageDraw.Draw(mask1)
            draw1.polygon([(0, 0), (cw, 0), (0, ch)], fill=255)

            tri1 = Image.new("RGB", (cw, ch), (0, 0, 0))
            tri1.paste(cell, (0, 0), mask1)
            triangles.append((tri1, mask1))

            # Üçgen 2 maskesi
            mask2 = Image.new("L", (cw, ch), 0)
            draw2 = ImageDraw.Draw(mask2)
            draw2.polygon([(cw, 0), (cw, ch), (0, ch)], fill=255)

            tri2 = Image.new("RGB", (cw, ch), (0, 0, 0))
            tri2.paste(cell, (0, 0), mask2)
            triangles.append((tri2, mask2))

    random.shuffle(triangles)

    out = Image.new("RGB", (w, h), (0, 0, 0))

    i = 0
    for y in range(grid):
        for x in range(grid):
            x0 = x * cell_w
            y0 = y * cell_h

            tri1, mask1 = triangles[i]
            i += 1
            tri2, mask2 = triangles[i]
            i += 1

            out.paste(tri1, (x0, y0), mask1)
            out.paste(tri2, (x0, y0), mask2)

    return out


def ligne_claire(im):
    im = im.convert("RGB")

    # Renkleri sadeleştir
    base = ImageEnhance.Color(im).enhance(1.4)
    base = ImageEnhance.Contrast(base).enhance(1.2)
    base = ImageOps.posterize(base, bits=4)
    base = base.filter(ImageFilter.SMOOTH_MORE)

    # Temiz siyah çizgiler
    gray = im.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.invert(edges)

    # Çizgileri sertleştir
    edges = edges.point(lambda x: 255 if x > 150 else 0)

    # Çizgileri biraz kalınlaştır
    edges = edges.filter(ImageFilter.MinFilter(3))
    edges = edges.convert("RGB")

    return ImageChops.multiply(base, edges)

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



def random_region_sudoku(im, region_size=0.5, grid=4):
    im = im.convert("RGB")
    w, h = im.size

    rw = int(w * region_size)
    rh = int(h * region_size)

    x0 = random.randint(0, w - rw)
    y0 = random.randint(0, h - rh)

    region = im.crop((x0, y0, x0 + rw, y0 + rh))

    cell_w = rw // grid
    cell_h = rh // grid

    pieces = []

    for y in range(grid):
        for x in range(grid):
            box = (
                x * cell_w,
                y * cell_h,
                (x + 1) * cell_w if x < grid - 1 else rw,
                (y + 1) * cell_h if y < grid - 1 else rh
            )
            pieces.append(region.crop(box))

    random.shuffle(pieces)

    shuffled = Image.new("RGB", (rw, rh))

    i = 0
    for y in range(grid):
        for x in range(grid):
            shuffled.paste(pieces[i], (x * cell_w, y * cell_h))
            i += 1

    out = im.copy()
    out.paste(shuffled, (x0, y0))

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
