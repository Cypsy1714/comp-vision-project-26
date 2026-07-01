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

    if v == "art_nouveau":
        im = im.convert("RGB")

        # Yumuşatılmış temel görüntü
        base = im.filter(ImageFilter.SMOOTH_MORE)
        base = base.filter(ImageFilter.GaussianBlur(radius=1.2))

        # Pastel / posterize renkler
        base = ImageOps.posterize(base, bits=4)
        base = ImageEnhance.Color(base).enhance(0.75)
        base = ImageEnhance.Contrast(base).enhance(1.15)
        base = ImageEnhance.Brightness(base).enhance(1.05)

        # Kenar çizgileri
        gray = im.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.autocontrast(edges)
        edges = ImageOps.invert(edges)

        # Kenarları sıcak kahverengi / altın tonuna çevir
        edge_rgb = ImageOps.colorize(
            edges,
            black=(80, 45, 20),
            white=(255, 230, 170)
        )

        # Kenarları görüntüyle harmanla
        out = ImageChops.multiply(base, edge_rgb)

        # Dekoratif kıvrımlı çizgiler
        draw = ImageDraw.Draw(out)
        w, h = out.size

        for _ in range(random.randint(8, 16)):
            points = []

            start_x = random.randint(0, w)
            start_y = random.randint(0, h)

            amp = random.randint(20, 70)
            length = random.randint(w // 4, w)
            step = random.randint(12, 25)

            for i in range(0, length, step):
                x = start_x + i
                y = start_y + int(math.sin(i / 30) * amp)

                if 0 <= x < w and 0 <= y < h:
                    points.append((x, y))

            if len(points) > 2:
                color = random.choice([
                    (90, 55, 25),
                    (120, 75, 35),
                    (160, 110, 55),
                    (70, 90, 60)
                ])

                draw.line(
                    points,
                    fill=color,
                    width=random.randint(1, 3)
                )

        # Hafif sepia sıcaklığı
        overlay = Image.new("RGB", out.size, (255, 235, 190))
        out = Image.blend(out, overlay, 0.12)

        return out
    

    
    
    if v == "spatter":
        im = im.convert("RGB")
        w, h = im.size

        out = im.copy()
        draw = ImageDraw.Draw(out, "RGBA")

        # Kaç tane ana leke olacak
        n_blobs = random.randint(80, 180)

        # Çamur / boya renkleri
        colors = [
            (60, 40, 20, 120),
            (80, 55, 30, 140),
            (30, 30, 30, 110),
            (255, 255, 255, 90)   # su damlası gibi
        ]

        for _ in range(n_blobs):

            cx = random.randint(0, w)
            cy = random.randint(0, h)

            base_radius = random.randint(4, 18)
            color = random.choice(colors)

            # Bir lekeyi birçok küçük daireyle oluştur
            for _ in range(random.randint(8, 25)):

                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(0, base_radius * 2)

                x = cx + math.cos(angle) * dist
                y = cy + math.sin(angle) * dist

                r = random.randint(1, max(2, base_radius // 2))

                draw.ellipse(
                    [
                        x - r,
                        y - r,
                        x + r,
                        y + r
                    ],
                    fill=color
                )

        # soften a bit
        out = out.filter(ImageFilter.GaussianBlur(0.8))

        return out


    if v == "bauhaus":
        im = im.convert("RGB")
        w, h = im.size

        # simplefy the photo
        base = ImageOps.posterize(im, bits=3)
        base = ImageEnhance.Color(base).enhance(0.6)
        base = ImageEnhance.Contrast(base).enhance(1.25)

        # Açık krem arka planla hafif karıştır
        bg = Image.new("RGB", (w, h), (235, 226, 205))
        out = Image.blend(base, bg, 0.35)

        draw = ImageDraw.Draw(out, "RGBA")

        bauhaus_colors = [
            (220, 40, 35, 150),    # kırmızı
            (245, 200, 35, 150),   # sarı
            (25, 80, 180, 150),    # mavi
            (20, 20, 20, 170),     # siyah
            (245, 245, 235, 130)   # beyaz/krem
        ]

        # Büyük geometrik formlar
        for _ in range(random.randint(8, 16)):
            shape = random.choice(["circle", "rect", "line", "triangle"])

            x = random.randint(0, w)
            y = random.randint(0, h)
            size = random.randint(min(w, h) // 10, min(w, h) // 3)
            color = random.choice(bauhaus_colors)

            if shape == "circle":
                draw.ellipse(
                    [x - size, y - size, x + size, y + size],
                    fill=color,
                    outline=(20, 20, 20, 180),
                    width=random.randint(2, 5)
                )

            elif shape == "rect":
                draw.rectangle(
                    [x, y, x + size, y + size],
                    fill=color,
                    outline=(20, 20, 20, 180),
                    width=random.randint(2, 5)
                )

            elif shape == "line":
                x2 = x + random.randint(-size, size)
                y2 = y + random.randint(-size, size)

                draw.line(
                    [x, y, x2, y2],
                    fill=(20, 20, 20, 200),
                    width=random.randint(4, 10)
                )

            elif shape == "triangle":
                points = [
                    (x, y - size),
                    (x - size, y + size),
                    (x + size, y + size)
                ]

                draw.polygon(
                    points,
                    fill=color,
                    outline=(20, 20, 20, 180)
                )

        # Bauhaus afiş hissi için ince grid çizgileri
        grid = random.randint(60, 120)

        for gx in range(0, w, grid):
            draw.line(
                [(gx, 0), (gx, h)],
                fill=(20, 20, 20, 35),
                width=1
            )

        for gy in range(0, h, grid):
            draw.line(
                [(0, gy), (w, gy)],
                fill=(20, 20, 20, 35),
                width=1
            )

        return out


    if v == "grid_dropout":
        im = im.convert("RGB")
        out = im.copy()
        draw = ImageDraw.Draw(out)

        w, h = out.size

        grid_size = random.randint(20, 60)
        drop_prob = random.uniform(0.25, 0.55)

        fill_mode = random.choice(["black", "white", "gray", "random"])

        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                if random.random() < drop_prob:
                    if fill_mode == "black":
                        color = (0, 0, 0)
                    elif fill_mode == "white":
                        color = (255, 255, 255)
                    elif fill_mode == "gray":
                        g = random.randint(40, 220)
                        color = (g, g, g)
                    else:
                        color = (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255)
                        )

                    x2 = min(x + grid_size, w)
                    y2 = min(y + grid_size, h)

                    draw.rectangle([x, y, x2, y2], fill=color)

        return out





    

    if v == "random_shape_occluder":
        im = im.convert("RGB")
        out = im.copy()
        draw = ImageDraw.Draw(out)

        w, h = out.size
        num_shapes = random.randint(8, 20)

        for _ in range(num_shapes):
            shape_type = random.choice(["rect", "ellipse", "triangle", "polygon"])

            cx = random.randint(0, w)
            cy = random.randint(0, h)

            size = random.randint(
                max(20, min(w, h) // 15),
                max(40, min(w, h) // 5)
            )

            color_mode = random.choice(["black", "white", "gray", "random"])

            if color_mode == "black":
                color = (0, 0, 0)
            elif color_mode == "white":
                color = (255, 255, 255)
            elif color_mode == "gray":
                g = random.randint(40, 220)
                color = (g, g, g)
            else:
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )

            if shape_type == "rect":
                x1 = cx - size
                y1 = cy - size
                x2 = cx + size
                y2 = cy + size
                draw.rectangle([x1, y1, x2, y2], fill=color)

            elif shape_type == "ellipse":
                x1 = cx - size
                y1 = cy - size
                x2 = cx + size
                y2 = cy + size
                draw.ellipse([x1, y1, x2, y2], fill=color)

            elif shape_type == "triangle":
                points = [
                    (cx, cy - size),
                    (cx - size, cy + size),
                    (cx + size, cy + size)
                ]
                draw.polygon(points, fill=color)

            elif shape_type == "polygon":
                sides = random.randint(5, 8)
                points = []

                for i in range(sides):
                    angle = 2 * math.pi * i / sides
                    r = random.randint(size // 2, size)
                    x = cx + int(math.cos(angle) * r)
                    y = cy + int(math.sin(angle) * r)
                    points.append((x, y))

                draw.polygon(points, fill=color)

        return out

    
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
    

    if v == "superman_laser_eyes":
        im = im.convert("RGB")
        out = im.copy()
        draw = ImageDraw.Draw(out, "RGBA")

        w, h = out.size

        # Basit varsayım: gözler görüntünün üst-orta kısmında
        eye_y = int(h * 0.38)
        left_eye_x = int(w * 0.42)
        right_eye_x = int(w * 0.58)

        laser_length = int(w * 0.9)

        for eye_x in [left_eye_x, right_eye_x]:
            end_x = eye_x + laser_length
            end_y = eye_y + random.randint(-25, 25)

            # Dış glow
            for width, alpha in [(26, 50), (18, 80), (10, 130)]:
                draw.line(
                    [(eye_x, eye_y), (end_x, end_y)],
                    fill=(255, 0, 0, alpha),
                    width=width
                )

            # Ana lazer
            draw.line(
                [(eye_x, eye_y), (end_x, end_y)],
                fill=(255, 20, 20, 230),
                width=5
            )

            # Beyaz sıcak merkez
            draw.line(
                [(eye_x, eye_y), (end_x, end_y)],
                fill=(255, 230, 230, 180),
                width=2
            )

            # Göz parlaması
            draw.ellipse(
                [eye_x - 12, eye_y - 12, eye_x + 12, eye_y + 12],
                fill=(255, 0, 0, 160)
            )

        return out

    if v == "frost":
        im = im.convert("RGB")
        w, h = im.size

        arr = np.array(im).astype(np.float32)

        # Buzlu cam bozulması
        out_arr = arr.copy()
        radius = random.randint(3, 8)

        for y in range(h):
            for x in range(w):
                dx = random.randint(-radius, radius)
                dy = random.randint(-radius, radius)

                nx = min(max(x + dx, 0), w - 1)
                ny = min(max(y + dy, 0), h - 1)

                out_arr[y, x] = arr[ny, nx]

        out = Image.fromarray(np.uint8(out_arr))

        # Soğuk mavi ton
        blue_overlay = Image.new("RGB", (w, h), (180, 220, 255))
        out = Image.blend(out, blue_overlay, 0.22)

        # Kontrastı biraz düşür, parlaklığı artır
        out = ImageEnhance.Contrast(out).enhance(0.85)
        out = ImageEnhance.Brightness(out).enhance(1.08)

        # Buz kristali / beyaz çizik efekti
        draw = ImageDraw.Draw(out, "RGBA")

        for _ in range(random.randint(80, 160)):
            x = random.randint(0, w)
            y = random.randint(0, h)
            length = random.randint(10, 45)
            angle = random.uniform(0, math.pi)

            x2 = x + int(math.cos(angle) * length)
            y2 = y + int(math.sin(angle) * length)

            draw.line(
                [(x, y), (x2, y2)],
                fill=(255, 255, 255, random.randint(50, 120)),
                width=random.randint(1, 2)
            )

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
