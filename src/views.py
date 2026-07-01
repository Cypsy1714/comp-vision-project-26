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



    if v == "affine_translate":
        im = im.convert("RGB")
        w, h = im.size

        shift_x = random.randint(-w // 6, w // 6)
        shift_y = random.randint(-h // 7, h // 7)

        out = im.transform(
            (w, h),
            Image.AFFINE,
            (1, 0, shift_x, 0, 1, shift_y),
            resample=Image.BICUBIC,
            fillcolor=(0, 0, 0)
        )   

        return out




    if v == "opposite_colors":
        im = im.convert("RGB")

        arr = np.array(im).astype(np.uint8)

        # Her kanalın tamamlayıcı (complementary) rengi
        arr = 255 - arr

        out = Image.fromarray(arr)

        return out


    if v == "complementary_colors":

        im = im.convert("RGB")
        arr = np.array(im).astype(np.float32) / 255.0

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        delta = maxc - minc

        h = np.zeros_like(maxc)
        s = np.zeros_like(maxc)
        v_channel = maxc

        mask = delta != 0

        s[maxc != 0] = delta[maxc != 0] / maxc[maxc != 0]

        idx = (maxc == r) & mask
        h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6

        idx = (maxc == g) & mask
        h[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2

        idx = (maxc == b) & mask
        h[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4

        h /= 6.0

        # Hue'yu 180° döndür
        h = (h + 0.5) % 1.0

        i = np.floor(h * 6).astype(int)
        f = h * 6 - i

        p = v_channel * (1 - s)
        q = v_channel * (1 - f * s)
        t = v_channel * (1 - (1 - f) * s)

        r2 = np.zeros_like(h)
        g2 = np.zeros_like(h)
        b2 = np.zeros_like(h)

        i = i % 6

        m = i == 0
        r2[m], g2[m], b2[m] = v_channel[m], t[m], p[m]

        m = i == 1
        r2[m], g2[m], b2[m] = q[m], v_channel[m], p[m]

        m = i == 2
        r2[m], g2[m], b2[m] = p[m], v_channel[m], t[m]

        m = i == 3
        r2[m], g2[m], b2[m] = p[m], q[m], v_channel[m]

        m = i == 4
        r2[m], g2[m], b2[m] = t[m], p[m], v_channel[m]

        m = i == 5
        r2[m], g2[m], b2[m] = v_channel[m], p[m], q[m]

        out = np.stack([r2, g2, b2], axis=2)
        out = np.clip(out * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(out)


    if v == "opposite_complementary_shuffle":
        im = im.convert("RGB")
        w, h = im.size

        # 1) Opposite colors
        opposite = ImageOps.invert(im)

        # 2) Complementary colors - HSV hue 180 derece çevirme
        arr = np.array(im).astype(np.float32) / 255.0

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        delta = maxc - minc

        hue = np.zeros_like(maxc)
        sat = np.zeros_like(maxc)
        val = maxc

        mask = delta != 0

        sat[maxc != 0] = delta[maxc != 0] / maxc[maxc != 0]

        idx = (maxc == r) & mask
        hue[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6

        idx = (maxc == g) & mask
        hue[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2

        idx = (maxc == b) & mask
        hue[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4

        hue /= 6.0
        hue = (hue + 0.5) % 1.0

        i = np.floor(hue * 6).astype(int) % 6
        f = hue * 6 - np.floor(hue * 6)

        p = val * (1 - sat)
        q = val * (1 - f * sat)
        t = val * (1 - (1 - f) * sat)

        r2 = np.zeros_like(hue)
        g2 = np.zeros_like(hue)
        b2 = np.zeros_like(hue)

        m = i == 0
        r2[m], g2[m], b2[m] = val[m], t[m], p[m]

        m = i == 1
        r2[m], g2[m], b2[m] = q[m], val[m], p[m]

        m = i == 2
        r2[m], g2[m], b2[m] = p[m], val[m], t[m]

        m = i == 3
        r2[m], g2[m], b2[m] = p[m], q[m], val[m]

        m = i == 4
        r2[m], g2[m], b2[m] = t[m], p[m], val[m]

        m = i == 5
        r2[m], g2[m], b2[m] = val[m], p[m], q[m]

        comp_arr = np.stack([r2, g2, b2], axis=2)
        comp_arr = np.clip(comp_arr * 255, 0, 255).astype(np.uint8)
        complementary = Image.fromarray(comp_arr)

        # 3) Fotoğrafın yarısı opposite, yarısı complementary
        mixed = Image.new("RGB", (w, h))

        if random.choice([True, False]):
            # Sol yarı opposite, sağ yarı complementary
            mixed.paste(opposite.crop((0, 0, w // 2, h)), (0, 0))
            mixed.paste(complementary.crop((w // 2, 0, w, h)), (w // 2, 0))
        else:
            # Üst yarı opposite, alt yarı complementary
            mixed.paste(opposite.crop((0, 0, w, h // 2)), (0, 0))
            mixed.paste(complementary.crop((0, h // 2, w, h)), (0, h // 2))

        # 4) Dikey fotoğrafsa 8 yatay parça, yatay fotoğrafsa 8 dikey parça
        pieces = []

        if h > w:
            # Dikey fotoğraf: 8 eşit yatay şerit
            piece_h = h // 8

            for i in range(8):
                y1 = i * piece_h
                y2 = h if i == 7 else (i + 1) * piece_h

                piece = mixed.crop((0, y1, w, y2))
                pieces.append(piece)

            random.shuffle(pieces)

            out = Image.new("RGB", (w, h))
            current_y = 0

            for piece in pieces:
                out.paste(piece, (0, current_y))
                current_y += piece.size[1]

        else:
            # Yatay fotoğraf: 8 eşit dikey şerit
            piece_w = w // 8

            for i in range(8):
                x1 = i * piece_w
                x2 = w if i == 7 else (i + 1) * piece_w

                piece = mixed.crop((x1, 0, x2, h))
                pieces.append(piece)

            random.shuffle(pieces)

            out = Image.new("RGB", (w, h))
            current_x = 0

            for piece in pieces:
                out.paste(piece, (current_x, 0))
                current_x += piece.size[0]

        return out





    if v == "sudoku_quarter_mix":
        im = im.convert("RGB")
        w, h = im.size

        # 4 main effects
        gray = ImageOps.grayscale(im).convert("RGB")

        sobel_gray = ImageOps.grayscale(im)
        sobel = sobel_gray.filter(ImageFilter.FIND_EDGES)
        sobel = ImageOps.autocontrast(sobel).convert("RGB")

        scale = random.uniform(1.25, 1.75)
        sw, sh = int(w * scale), int(h * scale)
        scaled = im.resize((sw, sh), Image.Resampling.BICUBIC)
        left = (sw - w) // 2
        top = (sh - h) // 2
        affine_scaled = scaled.crop((left, top, left + w, top + h))

        orange_overlay = Image.new("RGB", (w, h), (255, 125, 20))
        orange = Image.blend(im, orange_overlay, 0.45)
        orange = ImageEnhance.Color(orange).enhance(1.4)
        orange = ImageEnhance.Contrast(orange).enhance(1.1)

        variants = [gray, sobel, affine_scaled, orange]

        # Sudoku grid
        grid_n = random.choice([4, 6, 8])
        tile_w = w // grid_n
        tile_h = h // grid_n

        out = Image.new("RGB", (w, h))

        # every effect 1/4
        cells = list(range(grid_n * grid_n))
        random.shuffle(cells)

        groups = [
            cells[0::4],
            cells[1::4],
            cells[2::4],
            cells[3::4]
        ]

        cell_to_effect = {}

        for effect_id, group in enumerate(groups):
            for cell in group:
                cell_to_effect[cell] = effect_id

        for row in range(grid_n):
            for col in range(grid_n):
                cell_id = row * grid_n + col
                effect_id = cell_to_effect[cell_id]

                x1 = col * tile_w
                y1 = row * tile_h

                x2 = w if col == grid_n - 1 else x1 + tile_w
                y2 = h if row == grid_n - 1 else y1 + tile_h

                tile = variants[effect_id].crop((x1, y1, x2, y2))
                out.paste(tile, (x1, y1))

        # sudoku feeling
        draw = ImageDraw.Draw(out, "RGBA")

        for x in range(0, w, tile_w):
            draw.line([(x, 0), (x, h)], fill=(0, 0, 0, 50), width=1)

        for y in range(0, h, tile_h):
            draw.line([(0, y), (w, y)], fill=(0, 0, 0, 50), width=1)

        return out


    if v == "red_green_duotone":

        gray = ImageOps.grayscale(im)

        out = ImageOps.colorize(
            gray,
            black=(180, 0, 0),      # koyular kırmızı
            white=(0, 220, 0)       # açıklar yeşil
        )

        return out



    if v == "red_green_grid":

        gray = ImageOps.grayscale(im)

        out = ImageOps.colorize(
            gray,
            black=(180, 0, 0),
            white=(0, 220, 0)
        )

        out = ImageEnhance.Color(out).enhance(1.4)
        out = ImageEnhance.Contrast(out).enhance(1.2)

        draw = ImageDraw.Draw(out, "RGBA")

        w, h = out.size

        # Rastgele grid aralığı
        grid = random.randint(25, 80)

        # Rastgele çizgi kalınlığı
        thickness = random.randint(1, 4)

        # Grid rengi
        grid_color = random.choice([
            (255, 255, 255, 70),
            (0, 0, 0, 70),
            (255, 0, 0, 80),
            (0, 255, 0, 80)
        ])

        # Dikey çizgiler
        for x in range(0, w, grid):
            draw.line(
                [(x, 0), (x, h)],
                fill=grid_color,
                width=thickness
            )

        # Yatay çizgiler
        for y in range(0, h, grid):
            draw.line(
                [(0, y), (w, y)],
                fill=grid_color,
                width=thickness
            )

        return out

    if v == "affine_scale":
        im = im.convert("RGB")
        w, h = im.size

        # Güçlü ölçekleme
        scale = random.uniform(0.45, 1.75)

        new_w = int(w * scale)
        new_h = int(h * scale)

        scaled = im.resize((new_w, new_h), Image.Resampling.BICUBIC)

        canvas = Image.new("RGB", (w, h), (0, 0, 0))

        x = (w - new_w) // 2
        y = (h - new_h) // 2

        canvas.paste(scaled, (x, y))

        # Zoom in ise ortadan crop al
        if scale > 1:
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            canvas = scaled.crop((left, top, left + w, top + h))

        return canvas


    if v == "red_green_grid_mirror_red_restore":

        im = im.convert("RGB")
        w, h = im.size

        # Orijinal görüntüyü sakla
        original = im.copy()

        # Red-green duotone
        gray = ImageOps.grayscale(im)

        duotone = ImageOps.colorize(
            gray,
            black=(180, 0, 0),
            white=(0, 220, 0)
        )

        duotone = ImageEnhance.Color(duotone).enhance(1.4)
        duotone = ImageEnhance.Contrast(duotone).enhance(1.2)

        out = duotone.copy()

        # Rastgele grid
        grid = random.randint(35, 80)

        for y in range(0, h, grid):
            for x in range(0, w, grid):

                x2 = min(x + grid, w)
                y2 = min(y + grid, h)

                tile = duotone.crop((x, y, x2, y2))
                tile_arr = np.array(tile).astype(np.float32)

                r_mean = tile_arr[:, :, 0].mean()
                g_mean = tile_arr[:, :, 1].mean()

                # Karede kırmızı baskınsa:
                if r_mean > g_mean:
                    original_tile = original.crop((x, y, x2, y2))

                    # Aynala
                    original_tile = ImageOps.mirror(original_tile)

                    # Normal rengine geri döndürerek yapıştır
                    out.paste(original_tile, (x, y))

        # Grid çizgilerini en son ekle
        draw = ImageDraw.Draw(out, "RGBA")

        grid_color = random.choice([
            (255, 255, 255, 70),
            (0, 0, 0, 80),
            (255, 0, 0, 90),
            (0, 255, 0, 90)
        ])

        thickness = random.randint(1, 3)

        for gx in range(0, w, grid):
            draw.line(
                [(gx, 0), (gx, h)],
                fill=grid_color,
                width=thickness
            )

        for gy in range(0, h, grid):
            draw.line(
                [(0, gy), (w, gy)],
                fill=grid_color,
                width=thickness
            )

        return out


    if v == "anime":

        im = im.convert("RGB")

        # Smooth colors while preserving large structures
        base = im.filter(ImageFilter.MedianFilter(size=5))
        base = base.filter(ImageFilter.SMOOTH_MORE)
        base = base.filter(ImageFilter.GaussianBlur(radius=0.6))

        # Reduce the number of colors
        base = ImageOps.posterize(base, bits=5)

        # Boost saturation and contrast
        base = ImageEnhance.Color(base).enhance(2.2)
        base = ImageEnhance.Contrast(base).enhance(1.35)
        base = ImageEnhance.Sharpness(base).enhance(1.8)

        # Generate black outlines
        gray = ImageOps.grayscale(im)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.autocontrast(edges)

        # Make outlines thicker
        edges = edges.filter(ImageFilter.MaxFilter(3))

        edge_arr = np.array(edges)
        edge_mask = edge_arr > 40

        edge_img = Image.fromarray(
            np.where(edge_mask, 0, 255).astype(np.uint8)
        ).convert("RGB")

        # Combine outlines with colors
        out = ImageChops.multiply(base, edge_img)

        # Slight brightness boost
        out = ImageEnhance.Brightness(out).enhance(1.05)

        return out


    if v == "advanced_anime":

        im = im.convert("RGB")
        w, h = im.size

        # Smooth the image while keeping major shapes
        base = im.filter(ImageFilter.MedianFilter(size=5))
        base = base.filter(ImageFilter.SMOOTH_MORE)
        base = base.filter(ImageFilter.GaussianBlur(radius=0.8))

        arr = np.array(base).astype(np.float32)

        # Color quantization for cel-shading look
        levels = random.choice([5, 6, 7])
        arr = np.floor(arr / (256 / levels)) * (256 / levels)
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        cel = Image.fromarray(arr)

        # Strong anime-like saturation and contrast
        cel = ImageEnhance.Color(cel).enhance(2.0)
        cel = ImageEnhance.Contrast(cel).enhance(1.35)
        cel = ImageEnhance.Brightness(cel).enhance(1.08)

        # Create clean black outlines
        gray = ImageOps.grayscale(im)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.autocontrast(edges)
        edges = edges.filter(ImageFilter.MaxFilter(3))

        edge_arr = np.array(edges)
        edge_mask = edge_arr > random.randint(35, 55)

        outline = Image.fromarray(
            np.where(edge_mask, 0, 255).astype(np.uint8)
        ).convert("RGB")

        out = ImageChops.multiply(cel, outline)

        # Add soft bloom on bright areas
        bright = ImageOps.grayscale(out)
        bright_arr = np.array(bright)

        bloom_mask = np.where(bright_arr > 185, 255, 0).astype(np.uint8)
        bloom = Image.fromarray(bloom_mask).filter(ImageFilter.GaussianBlur(radius=6))
        bloom_rgb = ImageOps.colorize(
            bloom,
            black=(0, 0, 0),
            white=(255, 240, 220)
        )

        out = Image.blend(out, ImageChops.screen(out, bloom_rgb), 0.18)

        # Add slight blue-purple shadow tone
        shadow_overlay = Image.new("RGB", (w, h), (40, 55, 95))
        shadow_mask = ImageOps.grayscale(out)
        shadow_mask = ImageOps.invert(shadow_mask)
        shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=4))

        out = Image.composite(
            Image.blend(out, shadow_overlay, 0.18),
            out,
            shadow_mask
        )

        # Add small white eye/highlight sparkle-like points randomly
        draw = ImageDraw.Draw(out, "RGBA")

        for _ in range(random.randint(8, 20)):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            r = random.randint(1, 3)

            draw.ellipse(
                [x - r, y - r, x + r, y + r],
                fill=(255, 255, 255, random.randint(70, 140))
            )

        # Final sharpening
        out = ImageEnhance.Sharpness(out).enhance(1.6)

        return out



    if v == "random_yellow_frame":

        im = im.convert("RGB")
        out = im.copy()
        draw = ImageDraw.Draw(out, "RGBA")

        w, h = out.size

        # Pick a random scale while preserving aspect ratio
        scale = random.uniform(0.45, 0.95)

        frame_w = int(w * scale)
        frame_h = int(h * scale)

        # Center the frame
        x1 = (w - frame_w) // 2
        y1 = (h - frame_h) // 2
        x2 = x1 + frame_w
        y2 = y1 + frame_h

        # Random thick yellow border
        thickness = random.randint(
            max(4, min(w, h) // 80),
            max(10, min(w, h) // 25)
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(255, 220, 0, 255),
            width=thickness
        )

        return out

    if v == "random_thick_line":

        im = im.convert("RGB")
        out = im.copy()
        draw = ImageDraw.Draw(out, "RGBA")

        w, h = out.size

        # Pick a random start edge and end edge
        start_edge = random.choice(["top", "bottom", "left", "right"])
        end_edge = random.choice(["top", "bottom", "left", "right"])

        # Get a random point on a given edge
        def random_point_on_edge(edge):
            if edge == "top":
                return (random.randint(0, w), 0)
            if edge == "bottom":
                return (random.randint(0, w), h)
            if edge == "left":
                return (0, random.randint(0, h))
            if edge == "right":
                return (w, random.randint(0, h))

        p1 = random_point_on_edge(start_edge)
        p2 = random_point_on_edge(end_edge)

        # Random thick line style
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(160, 255)
        )

        thickness = random.randint(
            max(12, min(w, h) // 20),
            max(25, min(w, h) // 8)
        )

        # Draw glow first
        draw.line(
            [p1, p2],
            fill=(color[0], color[1], color[2], 70),
            width=thickness + random.randint(10, 30)
        )

        # Draw main thick line
        draw.line(
            [p1, p2],
            fill=color,
            width=thickness
        )

        return out

    if v == "mirror_wave":

        im = im.convert("RGB")

        # Önce aynala
        im = ImageOps.mirror(im)

        w, h = im.size
        arr = np.array(im)

        out = np.zeros_like(arr)

        amplitude = random.randint(10, 35)
        frequency = random.uniform(0.025, 0.075)

        # Yatay dalga: her satırı sağa-sola kaydırır
        for y in range(h):
            shift = int(math.sin(y * frequency) * amplitude)
            out[y] = np.roll(arr[y], shift, axis=0)

        return Image.fromarray(out)

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
