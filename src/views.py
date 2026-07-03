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
    


    if v == "center_shrink_fill_fragments":

        im = im.convert("RGB")
        w, h = im.size

        # Shrink the original image while preserving aspect ratio
        scale = random.uniform(0.45, 0.75)
        new_w = int(w * scale)
        new_h = int(h * scale)

        small = im.resize((new_w, new_h), Image.Resampling.BICUBIC)

        # Create empty canvas
        out = Image.new("RGB", (w, h))

        # Fill the whole background with random fragments from the image
        tile_size = random.randint(40, 120)

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):

                tw = min(tile_size, w - x)
                th = min(tile_size, h - y)

                sx = random.randint(0, max(0, w - tw))
                sy = random.randint(0, max(0, h - th))

                fragment = im.crop((sx, sy, sx + tw, sy + th))

                # Randomly flip some fragments
                if random.random() < 0.5:
                    fragment = ImageOps.mirror(fragment)

                if random.random() < 0.5:
                    fragment = ImageOps.flip(fragment)

                out.paste(fragment, (x, y))

        # Paste the shrunken original image in the center
        x0 = (w - new_w) // 2
        y0 = (h - new_h) // 2

        out.paste(small, (x0, y0))

        return out


    if v == "center_shrink_bauhaus_fill":

        im = im.convert("RGB")
        w, h = im.size

        # Shrink the original image while preserving aspect ratio
        scale = random.uniform(0.45, 0.72)
        new_w = int(w * scale)
        new_h = int(h * scale)

        small = im.resize((new_w, new_h), Image.Resampling.BICUBIC)

        # Create Bauhaus-style background
        out = Image.new("RGB", (w, h), (235, 226, 205))
        draw = ImageDraw.Draw(out, "RGBA")

        bauhaus_colors = [
            (220, 40, 35, 210),     # red
            (245, 200, 35, 210),    # yellow
            (25, 80, 180, 210),     # blue
            (20, 20, 20, 230),      # black
            (245, 245, 235, 190)    # cream
        ]

        # Draw random Bauhaus geometric shapes
        for _ in range(random.randint(18, 36)):

            shape = random.choice(["circle", "rect", "line", "triangle"])

            x = random.randint(-w // 8, w)
            y = random.randint(-h // 8, h)

            size = random.randint(
                max(20, min(w, h) // 12),
                max(50, min(w, h) // 4)
            )

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
                x2 = x + random.randint(-size * 2, size * 2)
                y2 = y + random.randint(-size * 2, size * 2)

                draw.line(
                    [(x, y), (x2, y2)],
                    fill=color,
                    width=random.randint(8, 22)
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

        # Add subtle Bauhaus grid lines
        grid = random.randint(55, 110)

        for gx in range(0, w, grid):
            draw.line(
                [(gx, 0), (gx, h)],
                fill=(20, 20, 20, 40),
                width=1
            )

        for gy in range(0, h, grid):
            draw.line(
                [(0, gy), (w, gy)],
                fill=(20, 20, 20, 40),
                width=1
            )

        # Paste the shrunken original image in the center
        x0 = (w - new_w) // 2
        y0 = (h - new_h) // 2

        # Add black frame behind the centered image
        frame = random.randint(8, 22)

        draw.rectangle(
            [x0 - frame, y0 - frame, x0 + new_w + frame, y0 + new_h + frame],
            fill=(20, 20, 20, 255)
        )

        out.paste(small, (x0, y0))

        return out


    if v == "center_bauhaus_fragments_fill":

        im = im.convert("RGB")
        w, h = im.size

        # Create a Bauhaus-style version of the original image
        bauhaus = ImageOps.posterize(im, bits=3)
        bauhaus = ImageEnhance.Color(bauhaus).enhance(0.65)
        bauhaus = ImageEnhance.Contrast(bauhaus).enhance(1.25)

        cream = Image.new("RGB", (w, h), (235, 226, 205))
        bauhaus = Image.blend(bauhaus, cream, 0.25)

        draw_b = ImageDraw.Draw(bauhaus, "RGBA")

        bauhaus_colors = [
            (220, 40, 35, 145),
            (245, 200, 35, 145),
            (25, 80, 180, 145),
            (20, 20, 20, 170),
            (245, 245, 235, 120)
        ]

        for _ in range(random.randint(8, 16)):

            shape = random.choice(["circle", "rect", "line", "triangle"])
            x = random.randint(0, w)
            y = random.randint(0, h)
            size = random.randint(min(w, h) // 10, min(w, h) // 3)
            color = random.choice(bauhaus_colors)

            if shape == "circle":
                draw_b.ellipse(
                    [x - size, y - size, x + size, y + size],
                    fill=color,
                    outline=(20, 20, 20, 170),
                    width=random.randint(2, 5)
                )

            elif shape == "rect":
                draw_b.rectangle(
                    [x, y, x + size, y + size],
                    fill=color,
                    outline=(20, 20, 20, 170),
                    width=random.randint(2, 5)
                )

            elif shape == "line":
                x2 = x + random.randint(-size, size)
                y2 = y + random.randint(-size, size)

                draw_b.line(
                    [(x, y), (x2, y2)],
                    fill=(20, 20, 20, 190),
                    width=random.randint(5, 12)
                )

            elif shape == "triangle":
                points = [
                    (x, y - size),
                    (x - size, y + size),
                    (x + size, y + size)
                ]

                draw_b.polygon(
                    points,
                    fill=color,
                    outline=(20, 20, 20, 170)
                )

        # Shrink the Bauhaus version while preserving aspect ratio
        scale = random.uniform(0.45, 0.72)
        new_w = int(w * scale)
        new_h = int(h * scale)

        small_bauhaus = bauhaus.resize((new_w, new_h), Image.Resampling.BICUBIC)

        # Fill the background with random fragments from the original image
        out = Image.new("RGB", (w, h))
        tile_size = random.randint(35, 110)

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):

                tw = min(tile_size, w - x)
                th = min(tile_size, h - y)

                sx = random.randint(0, max(0, w - tw))
                sy = random.randint(0, max(0, h - th))

                fragment = im.crop((sx, sy, sx + tw, sy + th))

                # Randomly rotate or mirror some fragments
                if random.random() < 0.5:
                    fragment = ImageOps.mirror(fragment)

                if random.random() < 0.5:
                    fragment = ImageOps.flip(fragment)

                out.paste(fragment, (x, y))

        # Paste the shrunken Bauhaus image in the center
        x0 = (w - new_w) // 2
        y0 = (h - new_h) // 2

        out.paste(small_bauhaus, (x0, y0))

        return out
    

    if v == "split_overlay_gradient_bg":

        im = im.convert("RGBA")
        w, h = im.size

        # Create a colorful gradient background
        c1 = np.array([
    random.randint(0, 255),
    random.randint(0, 255),
    random.randint(0, 255)
], dtype=np.float32)

        c2 = np.array([
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ], dtype=np.float32)

        c3 = np.array([
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ], dtype=np.float32)

        y = np.linspace(0, 1, h)[:, None, None]
        x = np.linspace(0, 1, w)[None, :, None]

        grad = (
            c1 * (1 - x) * (1 - y) +
            c2 * x * (1 - y) +
            c3 * y
        )

        grad = np.clip(grad, 0, 255).astype(np.uint8)
        bg = Image.fromarray(grad, "RGB").convert("RGBA")

        # Decide split direction
        if random.choice([True, False]):

            # Split image into left and right halves
            left = im.crop((0, 0, w // 2, h))
            right = im.crop((w // 2, 0, w, h))

            # Resize halves to full canvas height, keep their original half width
            left.putalpha(155)
            right.putalpha(155)

            # Place both halves on top of each other in the center
            x0 = (w - w // 2) // 2
            bg.alpha_composite(left, (x0, 0))
            bg.alpha_composite(right, (x0, 0))

        else:

            # Split image into top and bottom halves
            top = im.crop((0, 0, w, h // 2))
            bottom = im.crop((0, h // 2, w, h))

            top.putalpha(155)
            bottom.putalpha(155)

            # Place both halves on top of each other in the center
            y0 = (h - h // 2) // 2
            bg.alpha_composite(top, (0, y0))
            bg.alpha_composite(bottom, (0, y0))

        return bg.convert("RGB")


    if v == "upside_down":

        return im.transpose(Image.Transpose.ROTATE_180)


    if v == "gaussian_noise":

        im = im.convert("RGB")
        arr = np.array(im).astype(np.float32)

        # Generate Gaussian noise
        mean = 0
        sigma = random.uniform(15, 45)

        noise = np.random.normal(mean, sigma, arr.shape)

        # Add noise to the image
        noisy = arr + noise

        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy)


    if v == "upside_noise_wave":

        im = im.convert("RGB")

        # Rotate image upside down
        im = im.transpose(Image.Transpose.ROTATE_180)

        w, h = im.size
        arr = np.array(im).astype(np.float32)

        # Add strong Gaussian noise
        sigma = 65
        noise = np.random.normal(0, sigma, arr.shape)

        arr = arr + noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Apply fixed 5-wave horizontal distortion
        wave_arr = np.zeros_like(arr)

        amplitude = 28
        frequency = 2 * math.pi * 5 / h

        for y in range(h):
            shift = int(math.sin(y * frequency) * amplitude)
            wave_arr[y] = np.roll(arr[y], shift, axis=0)

        return Image.fromarray(wave_arr)
    

    if v == "multiplicative_noise":

        im = im.convert("RGB")
        arr = np.array(im).astype(np.float32)

        # Generate multiplicative Gaussian noise
        sigma = random.uniform(0.15, 0.45)
        noise = np.random.normal(0, sigma, arr.shape)

        # Apply multiplicative noise
        arr = arr * (1.0 + noise)

        arr = np.clip(arr, 0, 255).astype(np.uint8)

        return Image.fromarray(arr)
    

    if v == "neighbor_majority_20000":

        im = im.convert("RGB")

        # Resize image to approximately 4000 total pixels
        w, h = im.size
        total_pixels = 20000
        scale = math.sqrt(total_pixels / (w * h))

        small_w = max(2, int(w * scale))
        small_h = max(2, int(h * scale))

        small = im.resize((small_w, small_h), Image.Resampling.BILINEAR)

        arr = np.array(small)

        # Quantize colors so neighbor majority becomes meaningful
        levels = 50
        q = (arr // (256 // levels)) * (256 // levels)

        out = q.copy()

        # For every pixel, look at its 8 surrounding neighbors
        for y in range(small_h):
            for x in range(small_w):

                neighbors = []

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:

                        if dx == 0 and dy == 0:
                            continue

                        nx = x + dx
                        ny = y + dy

                        if 0 <= nx < small_w and 0 <= ny < small_h:
                            neighbors.append(tuple(q[ny, nx]))

                # Pick the most common neighboring color
                if neighbors:
                    most_common = max(set(neighbors), key=neighbors.count)
                    out[y, x] = most_common

        result = Image.fromarray(out.astype(np.uint8))

        # Scale back to original image size
        result = result.resize((w, h), Image.Resampling.NEAREST)

        return result
    

    if v == "neighbor_majority_25000_sudoku":

        im = im.convert("RGB")
        w, h = im.size

        # Resize image to approximately 25000 total pixels
        total_pixels = 25000
        scale = math.sqrt(total_pixels / (w * h))

        small_w = max(2, int(w * scale))
        small_h = max(2, int(h * scale))

        small = im.resize((small_w, small_h), Image.Resampling.BILINEAR)

        arr = np.array(small)

        # Quantize colors so majority voting works better
        levels = 100
        q = (arr // (256 // levels)) * (256 // levels)

        out = q.copy()

        # Apply 8-neighbor majority color filter
        for y in range(small_h):
            for x in range(small_w):

                neighbors = []

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:

                        if dx == 0 and dy == 0:
                            continue

                        nx = x + dx
                        ny = y + dy

                        if 0 <= nx < small_w and 0 <= ny < small_h:
                            neighbors.append(tuple(q[ny, nx]))

                if neighbors:
                    most_common = max(set(neighbors), key=neighbors.count)
                    out[y, x] = most_common

        result = Image.fromarray(out.astype(np.uint8))

        # Scale back to original image size
        result = result.resize((w, h), Image.Resampling.NEAREST)

        # Split into maximum 10 sudoku-like random pieces
        pieces_count = random.randint(4, 10)

        if h > w:
            # Vertical image: horizontal strips
            piece_h = h // pieces_count
            pieces = []

            for i in range(pieces_count):
                y1 = i * piece_h
                y2 = h if i == pieces_count - 1 else (i + 1) * piece_h
                pieces.append(result.crop((0, y1, w, y2)))

            random.shuffle(pieces)

            final = Image.new("RGB", (w, h))
            current_y = 0

            for piece in pieces:
                final.paste(piece, (0, current_y))
                current_y += piece.size[1]

        else:
            # Horizontal image: vertical strips
            piece_w = w // pieces_count
            pieces = []

            for i in range(pieces_count):
                x1 = i * piece_w
                x2 = w if i == pieces_count - 1 else (i + 1) * piece_w
                pieces.append(result.crop((x1, 0, x2, h)))

            random.shuffle(pieces)

            final = Image.new("RGB", (w, h))
            current_x = 0

            for piece in pieces:
                final.paste(piece, (current_x, 0))
                current_x += piece.size[0]

        return final


    if v == "wrap_roll":

        im = im.convert("RGB")
        w, h = im.size

        out = Image.new("RGB", (w, h))

        # Fill the background with random image fragments
        tile = random.randint(40, 100)

        for y in range(0, h, tile):
            for x in range(0, w, tile):

                tw = min(tile, w - x)
                th = min(tile, h - y)

                sx = random.randint(0, max(0, w - tw))
                sy = random.randint(0, max(0, h - th))

                fragment = im.crop((sx, sy, sx + tw, sy + th))

                if random.random() < 0.5:
                    fragment = ImageOps.mirror(fragment)

                if random.random() < 0.5:
                    fragment = fragment.rotate(
                        random.choice([90, 180, 270]),
                        expand=False
                    )

                out.paste(fragment, (x, y))

        # Roll the image from one side like a wrap
        direction = random.choice(["left", "right", "top", "bottom"])
        roll_ratio = random.uniform(0.35, 0.7)

        if direction in ["left", "right"]:

            strip_w = int(w * roll_ratio)

            if direction == "left":
                strip = im.crop((0, 0, strip_w, h))
                x0 = 0
            else:
                strip = im.crop((w - strip_w, 0, w, h))
                x0 = w - strip_w

            draw = ImageDraw.Draw(out, "RGBA")

            # Compress the strip progressively to imitate rolling
            steps = 28

            for i in range(steps):

                t = i / steps

                current_w = max(2, int(strip_w * (1 - t) ** 1.6))

                layer = strip.resize(
                    (current_w, h),
                    Image.Resampling.BICUBIC
                )

                if direction == "left":
                    xpos = x0 + i * strip_w // steps
                else:
                    xpos = x0 + strip_w - current_w - i * strip_w // steps

                out.paste(layer, (xpos, 0))

                # Draw a shadow to enhance the roll illusion
                draw.line(
                    [(xpos, 0), (xpos, h)],
                    fill=(0, 0, 0, 25),
                    width=3
                )

        else:

            strip_h = int(h * roll_ratio)

            if direction == "top":
                strip = im.crop((0, 0, w, strip_h))
                y0 = 0
            else:
                strip = im.crop((0, h - strip_h, w, h))
                y0 = h - strip_h

            draw = ImageDraw.Draw(out, "RGBA")

            steps = 28

            for i in range(steps):

                t = i / steps

                current_h = max(2, int(strip_h * (1 - t) ** 1.6))

                layer = strip.resize(
                    (w, current_h),
                    Image.Resampling.BICUBIC
                )

                if direction == "top":
                    ypos = y0 + i * strip_h // steps
                else:
                    ypos = y0 + strip_h - current_h - i * strip_h // steps

                out.paste(layer, (0, ypos))

                draw.line(
                    [(0, ypos), (w, ypos)],
                    fill=(0, 0, 0, 25),
                    width=3
                )

        return out


    if v == "mosaic_4_tile":

        im = im.convert("RGB")
        w, h = im.size

        # Create four smaller copies of the image
        tile_w = w // 2
        tile_h = h // 2

        tile = im.resize((tile_w, tile_h), Image.Resampling.BICUBIC)

        out = Image.new("RGB", (w, h))

        # Place the same image into a 2x2 mosaic
        out.paste(tile, (0, 0))
        out.paste(tile, (tile_w, 0))
        out.paste(tile, (0, tile_h))
        out.paste(tile, (tile_w, tile_h))

        return out


    if v == "mosaic_4_random_effects":

        im = im.convert("RGB")
        w, h = im.size

        tile_w = w // 2
        tile_h = h // 2

        effects = [
            "upside_down",
            "bauhaus",
            "opposite_colors",
            "sobel",
            "red_green_grid",
            "wave",
            "blackout",
            "center_shrink_bauhaus_fill",
            "horizontal_stretch"
        ]

        selected = random.sample(effects, 4)

        out = Image.new("RGB", (w, h))

        positions = [
            (0, 0),
            (tile_w, 0),
            (0, tile_h),
            (tile_w, tile_h)
        ]

        for i, effect in enumerate(selected):

            tile = im.resize((tile_w, tile_h), Image.Resampling.BICUBIC)

            # Apply a random selected effect to this tile
            tile = make(tile, effect)

            out.paste(tile, positions[i])

        return out


    if v == "cross_hatching":

        gray = ImageOps.grayscale(im)
        w, h = gray.size

        arr = np.array(gray)

        out = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(out)

        spacing = 8

        # First diagonal hatch
        for offset in range(-h, w, spacing):

            for t in range(max(0, -offset), min(h, w - offset)):

                x = offset + t
                y = t

                if arr[y, x] < 190:
                    draw.point((x, y), fill="black")

        # Second diagonal hatch
        for offset in range(0, w + h, spacing):

            for y in range(h):

                x = offset - y

                if 0 <= x < w:
                    if arr[y, x] < 140:
                        draw.point((x, y), fill="black")

        # Third hatch for darker areas
        for x in range(0, w, spacing):

            for y in range(h):

                if arr[y, x] < 90:
                    draw.point((x, y), fill="black")

        # Fourth hatch for the darkest regions
        for y in range(0, h, spacing):

            for x in range(w):

                if arr[y, x] < 45:
                    draw.point((x, y), fill="black")

        return out

    if v == "horizontal_stretch":

        im = im.convert("RGB")
        w, h = im.size

        arr = np.array(im)
        out = np.zeros_like(arr)

        # Stretch strength
        strength = random.uniform(0.8, 1.5)

        center = w / 2

        # Stretch horizontally from the center
        for x_out in range(w):

            t = (x_out - center) / center

            # Nonlinear stretching
            x_in = center + t * center / (1 + strength * (1 - abs(t)))

            x_in = np.clip(x_in, 0, w - 1)

            x0 = int(np.floor(x_in))
            x1 = min(x0 + 1, w - 1)

            alpha = x_in - x0

            out[:, x_out] = (
                (1 - alpha) * arr[:, x0] +
                alpha * arr[:, x1]
            )

        out = np.clip(out, 0, 255).astype(np.uint8)

        return Image.fromarray(out)

    if v == "snow":

        im = im.convert("RGB")
        w, h = im.size

        out = im.copy()
        draw = ImageDraw.Draw(out, "RGBA")

        # Slightly brighten and cool the image
        out = ImageEnhance.Brightness(out).enhance(1.08)
        out = ImageEnhance.Color(out).enhance(0.82)

        # Add soft blue-white overlay
        cold = Image.new("RGB", (w, h), (210, 230, 255))
        out = Image.blend(out, cold, 0.14)

        draw = ImageDraw.Draw(out, "RGBA")

        # Draw snowflakes with random sizes and opacity
        for _ in range(random.randint(400, 900)):

            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)

            r = random.choice([1, 1, 1, 2, 2, 3])
            alpha = random.randint(90, 220)

            draw.ellipse(
                [x - r, y - r, x + r, y + r],
                fill=(255, 255, 255, alpha)
            )

        # Add a few larger blurred snow blobs
        snow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        snow_draw = ImageDraw.Draw(snow_layer, "RGBA")

        for _ in range(random.randint(40, 90)):

            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)

            r = random.randint(3, 8)
            alpha = random.randint(45, 110)

            snow_draw.ellipse(
                [x - r, y - r, x + r, y + r],
                fill=(255, 255, 255, alpha)
            )

        snow_layer = snow_layer.filter(ImageFilter.GaussianBlur(radius=1.2))
        out = Image.alpha_composite(out.convert("RGBA"), snow_layer)

        return out.convert("RGB")


    if v == "rain_streaks":

        im = im.convert("RGB")
        w, h = im.size

        out = im.copy()

        # Darken and desaturate the image for rainy atmosphere
        out = ImageEnhance.Brightness(out).enhance(0.78)
        out = ImageEnhance.Color(out).enhance(0.65)
        out = ImageEnhance.Contrast(out).enhance(1.12)

        # Add a subtle cold blue overlay
        cold = Image.new("RGB", (w, h), (70, 95, 130))
        out = Image.blend(out, cold, 0.18)

        rain_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rain_layer, "RGBA")

        # Draw diagonal rain streaks
        angle = random.uniform(-0.45, -0.25)
        length = random.randint(18, 42)

        for _ in range(random.randint(500, 1100)):

            x = random.randint(-w // 5, w + w // 5)
            y = random.randint(-h // 5, h + h // 5)

            dx = int(math.cos(angle) * length)
            dy = int(math.sin(angle) * length)

            alpha = random.randint(55, 140)
            width = random.choice([1, 1, 1, 2])

            draw.line(
                [(x, y), (x + dx, y + dy)],
                fill=(210, 230, 255, alpha),
                width=width
            )

        # Slight blur makes rain look more motion-like
        rain_layer = rain_layer.filter(ImageFilter.GaussianBlur(radius=0.45))

        out = Image.alpha_composite(out.convert("RGBA"), rain_layer)

        return out.convert("RGB")


    if v == "color_quantization":

        im = im.convert("RGB")
        arr = np.array(im)

        # Reduce color levels per channel
        levels = random.choice([3, 4, 5, 6, 8])

        step = 256 // levels
        quantized = (arr // step) * step

        quantized = np.clip(quantized, 0, 255).astype(np.uint8)

        out = Image.fromarray(quantized)

        return out

    if v == "elastic_transform":

        im = im.convert("RGB")
        w, h = im.size

        arr = np.array(im)

        # Create random displacement fields
        alpha = random.uniform(25, 60)
        sigma = random.uniform(6, 12)

        dx = np.random.uniform(-1, 1, (h, w))
        dy = np.random.uniform(-1, 1, (h, w))

        dx_img = Image.fromarray(((dx + 1) * 127.5).astype(np.uint8))
        dy_img = Image.fromarray(((dy + 1) * 127.5).astype(np.uint8))

        # Smooth displacement fields
        dx_img = dx_img.filter(ImageFilter.GaussianBlur(radius=sigma))
        dy_img = dy_img.filter(ImageFilter.GaussianBlur(radius=sigma))

        dx = (np.array(dx_img).astype(np.float32) / 127.5 - 1) * alpha
        dy = (np.array(dy_img).astype(np.float32) / 127.5 - 1) * alpha

        # Create coordinate grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)

        # Bilinear interpolation
        x0 = np.floor(map_x).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y0 = np.floor(map_y).astype(np.int32)
        y1 = np.clip(y0 + 1, 0, h - 1)

        wa = (x1 - map_x) * (y1 - map_y)
        wb = (x1 - map_x) * (map_y - y0)
        wc = (map_x - x0) * (y1 - map_y)
        wd = (map_x - x0) * (map_y - y0)

        out = (
            wa[..., None] * arr[y0, x0] +
            wb[..., None] * arr[y1, x0] +
            wc[..., None] * arr[y0, x1] +
            wd[..., None] * arr[y1, x1]
        )

        out = np.clip(out, 0, 255).astype(np.uint8)

        return Image.fromarray(out)



    if v == "drunk":

        im = im.convert("RGB")
        w, h = im.size

        # Create a soft astigmatism-like blur
        blur_x = im.filter(ImageFilter.GaussianBlur(radius=2.2))
        blur_y = im.resize((w, max(1, int(h * 0.96))), Image.Resampling.BICUBIC)
        blur_y = blur_y.resize((w, h), Image.Resampling.BICUBIC)
        blur_y = blur_y.filter(ImageFilter.GaussianBlur(radius=1.4))

        out = Image.blend(blur_x, blur_y, 0.45)

        # Create a second ghost image slightly shifted and transparent
        ghost = im.copy()
        ghost = ImageEnhance.Contrast(ghost).enhance(0.85)
        ghost = ImageEnhance.Brightness(ghost).enhance(1.05)

        shift_x = random.randint(-18, 18)
        shift_y = random.randint(-10, 10)

        ghost_layer = Image.new("RGB", (w, h), (0, 0, 0))
        ghost_layer.paste(ghost, (shift_x, shift_y))

        out = Image.blend(out, ghost_layer, 0.28)

        # Add slight color-channel separation
        arr = np.array(out)

        r = np.roll(arr[:, :, 0], random.randint(3, 8), axis=1)
        g = arr[:, :, 1]
        b = np.roll(arr[:, :, 2], random.randint(-8, -3), axis=1)

        arr = np.stack([r, g, b], axis=2)
        out = Image.fromarray(arr.astype(np.uint8))

        # Add mild wobble distortion
        arr = np.array(out)
        warped = np.zeros_like(arr)

        amplitude = random.randint(3, 8)
        frequency = random.uniform(0.025, 0.055)

        for y in range(h):
            shift = int(math.sin(y * frequency) * amplitude)
            warped[y] = np.roll(arr[y], shift, axis=0)

        out = Image.fromarray(warped)

        # Final softening
        out = out.filter(ImageFilter.GaussianBlur(radius=0.6))

        return out


    if v == "blackout":

        im = im.convert("RGB")
        w, h = im.size

        # Create a black canvas
        black = Image.new("RGB", (w, h), (0, 0, 0))

        # Create a grayscale mask
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        # Randomize eye opening size
        eye_w = int(w * random.uniform(0.22, 0.32))
        eye_h = int(h * random.uniform(0.22, 0.30))

        eye_y = int(h * random.uniform(0.48, 0.58))

        eye_distance = int(w * random.uniform(0.12, 0.18))

        left_x = w // 2 - eye_distance - eye_w // 2
        right_x = w // 2 + eye_distance - eye_w // 2

        # Draw the two eye openings
        draw.ellipse(
            (
                left_x,
                eye_y - eye_h // 2,
                left_x + eye_w,
                eye_y + eye_h // 2,
            ),
            fill=255,
        )

        draw.ellipse(
            (
                right_x,
                eye_y - eye_h // 2,
                right_x + eye_w,
                eye_y + eye_h // 2,
            ),
            fill=255,
        )

        # Slightly connect the two eyes
        bridge_h = int(eye_h * 0.25)
        draw.rectangle(
            (
                left_x + eye_w - eye_w // 8,
                eye_y - bridge_h // 2,
                right_x + eye_w // 8,
                eye_y + bridge_h // 2,
            ),
            fill=180,
        )

        # Heavy feathering for blackout effect
        blur_radius = random.randint(35, 70)
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

        # Composite original image with black background
        out = Image.composite(im, black, mask)

        # Slight blur to imitate fading vision
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # Mild chromatic aberration
        arr = np.array(out)

        r = np.roll(arr[:, :, 0], 2, axis=1)
        g = arr[:, :, 1]
        b = np.roll(arr[:, :, 2], -2, axis=1)

        out = Image.fromarray(np.stack([r, g, b], axis=2))

        return out


    if v == "post_impressionist":

        im = im.convert("RGB")
        w, h = im.size

        # Boost color and contrast for a vivid painted look
        base = ImageEnhance.Color(im).enhance(1.8)
        base = ImageEnhance.Contrast(base).enhance(1.25)
        base = ImageEnhance.Brightness(base).enhance(1.05)

        # Posterize colors to reduce photographic smoothness
        base = ImageOps.posterize(base, bits=5)

        out = base.copy()
        draw = ImageDraw.Draw(out, "RGBA")

        arr = np.array(base).astype(np.int16)

        # Brush stroke settings
        stroke_count = int((w * h) / 550)
        stroke_count = max(1200, min(stroke_count, 9000))

        for _ in range(stroke_count):

            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)

            color = arr[y, x]

            # Slight color variation per stroke
            color = (
                int(np.clip(color[0] + random.randint(-18, 18), 0, 255)),
                int(np.clip(color[1] + random.randint(-18, 18), 0, 255)),
                int(np.clip(color[2] + random.randint(-18, 18), 0, 255)),
                random.randint(40, 90)
            )

            length = random.randint(4, 10)
            width_s = random.randint(1, 3)

            # Direction changes with position for swirling painted movement
            angle = (
                math.sin(y * 0.035) * 1.2 +
                math.cos(x * 0.025) * 1.2 +
                random.uniform(-0.8, 0.8)
            )

            x2 = x + int(math.cos(angle) * length)
            y2 = y + int(math.sin(angle) * length)

            draw.line(
                [(x, y), (x2, y2)],
                fill=color,
                width=width_s
            )

        # Add darker short strokes for texture
        for _ in range(stroke_count // 3):

            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)

            color = arr[y, x]
            dark = (
                max(0, int(color[0] * 0.55)),
                max(0, int(color[1] * 0.55)),
                max(0, int(color[2] * 0.55)),
                random.randint(70, 130)
            )

            length = random.randint(4, 14)
            angle = random.uniform(0, 2 * math.pi)

            x2 = x + int(math.cos(angle) * length)
            y2 = y + int(math.sin(angle) * length)

            draw.line(
                [(x, y), (x2, y2)],
                fill=dark,
                width=random.randint(1, 3)
            )

        # Slight smoothing to merge strokes
        out = out.filter(ImageFilter.GaussianBlur(radius=0.35))

        # Final vivid color correction
        out = ImageEnhance.Color(out).enhance(1.25)
        out = ImageEnhance.Contrast(out).enhance(1.15)
        out = ImageEnhance.Sharpness(out).enhance(1.4)

        return out
    

    if v == "white_concrete_da_vinci":

        im = im.convert("RGB")
        w, h = im.size

        # Convert to grayscale to remove all original colors
        gray = ImageOps.grayscale(im)

        # Create soft sculptural base
        base = ImageOps.autocontrast(gray)
        base = ImageEnhance.Contrast(base).enhance(1.35)
        base = ImageEnhance.Brightness(base).enhance(1.25)

        # Map tones to white concrete / plaster colors
        statue = ImageOps.colorize(
            base,
            black=(105, 105, 100),
            white=(245, 242, 232)
        )

        # Create embossed relief effect
        embossed = gray.filter(ImageFilter.EMBOSS)
        embossed = ImageOps.autocontrast(embossed)
        embossed = ImageEnhance.Contrast(embossed).enhance(1.4)

        emboss_rgb = ImageOps.colorize(
            embossed,
            black=(80, 80, 75),
            white=(255, 252, 240)
        )

        statue = ImageChops.multiply(statue, emboss_rgb)

        # Add subtle concrete grain
        arr = np.array(statue).astype(np.float32)

        noise = np.random.normal(0, random.uniform(6, 14), arr.shape)
        arr = arr + noise

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)

        # Add carved dark edge details
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.autocontrast(edges)
        edges = edges.filter(ImageFilter.MaxFilter(3))

        edge_arr = np.array(edges)
        edge_mask = edge_arr > 35

        outline = Image.fromarray(
            np.where(edge_mask, 130, 255).astype(np.uint8)
        ).convert("RGB")

        out = ImageChops.multiply(out, outline)

        # Add slight blur and sharpen for stone-like surface
        out = out.filter(ImageFilter.GaussianBlur(radius=0.35))
        out = ImageEnhance.Sharpness(out).enhance(1.6)

        # Final white plaster brightness
        out = ImageEnhance.Brightness(out).enhance(1.12)
        out = ImageEnhance.Contrast(out).enhance(1.08)

        return out

    if v == "fenerbahce":

        im = im.convert("RGB")

        # Convert image to grayscale for tonal mapping
        gray = ImageOps.grayscale(im)

        # Inverted yellow-navy duotone:
        # Dark areas become yellow, bright areas become navy
        out = ImageOps.colorize(
            gray,
            black=(255, 220, 0),     # yellow
            white=(0, 35, 95)        # navy
        )

        # Increase contrast and saturation for a stronger Fenerbahce palette
        out = ImageEnhance.Contrast(out).enhance(1.25)
        out = ImageEnhance.Color(out).enhance(1.35)

        return out
    

    if v == "approx_living_fenerbahce_da_vinci_upside_down":

        im = im.convert("RGB")
        w, h = im.size

        # Create approximate living-area mask using skin-like and saturated warm colors
        arr = np.array(im).astype(np.float32)

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)

        saturation = (maxc - minc) / np.maximum(maxc, 1)

        # Approximate skin / warm organic tones
        skin_mask = (
            (r > 80) &
            (g > 35) &
            (b > 20) &
            (r > g * 1.08) &
            (r > b * 1.18) &
            ((r - b) > 18)
        )

        # Approximate colorful living/object regions
        saturated_mask = (
            (saturation > 0.28) &
            (maxc > 70)
        )

        mask_arr = np.where(skin_mask | saturated_mask, 255, 0).astype(np.uint8)

        mask = Image.fromarray(mask_arr)

        # Smooth and expand mask
        mask = mask.filter(ImageFilter.MaxFilter(9))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=5))

        # Fenerbahce version
        gray = ImageOps.grayscale(im)

        fenerbahce = ImageOps.colorize(
            gray,
            black=(255, 220, 0),
            white=(0, 35, 95)
        )

        fenerbahce = ImageEnhance.Contrast(fenerbahce).enhance(1.25)
        fenerbahce = ImageEnhance.Color(fenerbahce).enhance(1.35)

        # White concrete Da Vinci version
        base = ImageOps.autocontrast(gray)
        base = ImageEnhance.Contrast(base).enhance(1.35)
        base = ImageEnhance.Brightness(base).enhance(1.25)

        statue = ImageOps.colorize(
            base,
            black=(105, 105, 100),
            white=(245, 242, 232)
        )

        embossed = gray.filter(ImageFilter.EMBOSS)
        embossed = ImageOps.autocontrast(embossed)
        embossed = ImageEnhance.Contrast(embossed).enhance(1.4)

        emboss_rgb = ImageOps.colorize(
            embossed,
            black=(80, 80, 75),
            white=(255, 252, 240)
        )

        statue = ImageChops.multiply(statue, emboss_rgb)

        statue_arr = np.array(statue).astype(np.float32)

        # Add concrete-like grain
        noise = np.random.normal(0, random.uniform(6, 14), statue_arr.shape)
        statue_arr = statue_arr + noise
        statue_arr = np.clip(statue_arr, 0, 255).astype(np.uint8)

        da_vinci = Image.fromarray(statue_arr)

        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.autocontrast(edges)
        edges = edges.filter(ImageFilter.MaxFilter(3))

        edge_arr = np.array(edges)
        edge_mask = edge_arr > 35

        outline = Image.fromarray(
            np.where(edge_mask, 130, 255).astype(np.uint8)
        ).convert("RGB")

        da_vinci = ImageChops.multiply(da_vinci, outline)

        da_vinci = da_vinci.filter(ImageFilter.GaussianBlur(radius=0.35))
        da_vinci = ImageEnhance.Sharpness(da_vinci).enhance(1.6)
        da_vinci = ImageEnhance.Brightness(da_vinci).enhance(1.12)
        da_vinci = ImageEnhance.Contrast(da_vinci).enhance(1.08)

        # Composite approximate living areas over Da Vinci background
        out = Image.composite(fenerbahce, da_vinci, mask)

        # Rotate upside down
        out = out.transpose(Image.Transpose.ROTATE_180)

        return out


    if v == "golden_edges":

        im = im.convert("RGB")

        # Detect edges
        gray = ImageOps.grayscale(im)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.autocontrast(edges)

        # Keep only strong edges
        edge_arr = np.array(edges)
        mask = (edge_arr > 95).astype(np.uint8) * 255

        # Thicken the edges
        mask = Image.fromarray(mask)
        mask = mask.filter(ImageFilter.MaxFilter(3))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=0.35))

        # Create golden edge layer
        gold = Image.new("RGB", im.size, (150, 118, 45))

        # Composite gold only on edge mask
        out = Image.composite(gold, im, mask)

        return out


    if v == "outer_edges_random_value_fill":

        im = im.convert("RGB")

        # Detect stronger, cleaner edges
        gray = ImageOps.grayscale(im)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.autocontrast(edges)

        edge_arr = np.array(edges)

        # Higher threshold keeps fewer inner details
        mask = (edge_arr > 95).astype(np.uint8) * 255

        mask = Image.fromarray(mask)

        # Keep edges thin
        mask = mask.filter(ImageFilter.MaxFilter(3))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=0.35))

        # Create random RGB fill
        random_fill_arr = np.random.randint(
            0,
            256,
            (im.size[1], im.size[0], 3),
            dtype=np.uint8
        )

        random_fill = Image.fromarray(random_fill_arr, "RGB")

        # Replace detected edge areas with random values
        out = Image.composite(random_fill, im, mask)

        return out
    
    if v == "outer_edges_random_value_fill_wave_zoom":

        im = im.convert("RGB")

        # Random zoom
        zoom = random.uniform(1.05, 1.35)

        w, h = im.size
        zw = int(w * zoom)
        zh = int(h * zoom)

        enlarged = im.resize((zw, zh), Image.Resampling.BICUBIC)

        left = (zw - w) // 2
        top = (zh - h) // 2

        im = enlarged.crop((left, top, left + w, top + h))

        # Detect stronger, cleaner edges
        gray = ImageOps.grayscale(im)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.autocontrast(edges)

        edge_arr = np.array(edges)

        # Higher threshold keeps fewer inner details
        mask = (edge_arr > 95).astype(np.uint8) * 255

        mask = Image.fromarray(mask)
        mask = mask.filter(ImageFilter.MaxFilter(3))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=0.35))

        # Create random RGB fill
        random_fill = np.random.randint(
            0,
            256,
            (h, w, 3),
            dtype=np.uint8
        )

        random_fill = Image.fromarray(random_fill)

        # Replace edge areas with random values
        out = Image.composite(random_fill, im, mask)

        # Apply gentle wave distortion
        arr = np.array(out)
        warped = np.zeros_like(arr)

        amplitude = random.uniform(4, 10)
        frequency = 2 * math.pi * random.uniform(2.5, 4.5) / h

        for y in range(h):
            shift = int(math.sin(y * frequency) * amplitude)
            warped[y] = np.roll(arr[y], shift, axis=0)

        return Image.fromarray(warped)



    if v == "orange_noir_jazz":

        im = im.convert("RGB")
        w, h = im.size

        # Convert image to grayscale for controlled tonal mapping
        gray = ImageOps.grayscale(im)

        # Strong contrast for noir silhouettes
        gray = ImageOps.autocontrast(gray)
        gray = ImageEnhance.Contrast(gray).enhance(1.75)
        gray = ImageEnhance.Brightness(gray).enhance(0.82)

        # Map tones to dark brown / burnt orange / amber
        out = ImageOps.colorize(
            gray,
            black=(8, 5, 4),
            mid=(92, 42, 18),
            white=(245, 120, 22)
        )

        # Add warm orange glow
        glow = out.filter(ImageFilter.GaussianBlur(radius=8))
        orange = Image.new("RGB", (w, h), (255, 105, 12))
        glow = Image.blend(glow, orange, 0.22)
        out = Image.blend(out, glow, 0.18)

        # Add dark vignette
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        cx, cy = w / 2, h / 2
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        dist = dist / dist.max()

        vignette = 1 - np.clip((dist - 0.25) / 0.75, 0, 1) * 0.55

        arr = np.array(out).astype(np.float32)
        arr = arr * vignette[:, :, None]

        # Add subtle film grain
        noise = np.random.normal(0, 7, arr.shape)
        arr = arr + noise

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)

        # Final punchy noir correction
        out = ImageEnhance.Contrast(out).enhance(1.22)
        out = ImageEnhance.Color(out).enhance(1.15)

        return out
    

    if v == "orange_noir_expansion":

        im = im.convert("RGB")
        w, h = im.size

        # Apply orange noir jazz color grading
        gray = ImageOps.grayscale(im)
        gray = ImageOps.autocontrast(gray)
        gray = ImageEnhance.Contrast(gray).enhance(1.75)
        gray = ImageEnhance.Brightness(gray).enhance(0.82)

        colored = ImageOps.colorize(
            gray,
            black=(8, 5, 4),
            mid=(92, 42, 18),
            white=(245, 120, 22)
        )

        glow = colored.filter(ImageFilter.GaussianBlur(radius=8))
        orange = Image.new("RGB", (w, h), (255, 105, 12))
        glow = Image.blend(glow, orange, 0.22)
        colored = Image.blend(colored, glow, 0.18)

        # Pick random expansion center
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)

        arr = np.array(colored)
        out = np.zeros_like(arr)

        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        dx = xx - cx
        dy = yy - cy

        dist = np.sqrt(dx * dx + dy * dy)
        max_dist = np.sqrt(max(cx, w - cx) ** 2 + max(cy, h - cy) ** 2)

        norm = dist / max_dist

        # Expansion strength
        strength = random.uniform(0.25, 0.55)

        # Pull output pixels inward so the image appears to expand outward
        factor = 1 + strength * (1 - np.exp(-norm * 2.5))

        map_x = cx + dx / factor
        map_y = cy + dy / factor

        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)

        # Bilinear interpolation
        x0 = np.floor(map_x).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)

        y0 = np.floor(map_y).astype(np.int32)
        y1 = np.clip(y0 + 1, 0, h - 1)

        wx = map_x - x0
        wy = map_y - y0

        warped = (
            (1 - wx)[..., None] * (1 - wy)[..., None] * arr[y0, x0] +
            wx[..., None] * (1 - wy)[..., None] * arr[y0, x1] +
            (1 - wx)[..., None] * wy[..., None] * arr[y1, x0] +
            wx[..., None] * wy[..., None] * arr[y1, x1]
        )

        warped = np.clip(warped, 0, 255).astype(np.uint8)
        out = Image.fromarray(warped)

        # Add dark vignette after expansion
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        vcx, vcy = w / 2, h / 2
        vdist = np.sqrt((xx - vcx) ** 2 + (yy - vcy) ** 2)
        vdist = vdist / vdist.max()

        vignette = 1 - np.clip((vdist - 0.25) / 0.75, 0, 1) * 0.55

        out_arr = np.array(out).astype(np.float32)
        out_arr = out_arr * vignette[:, :, None]

        # Add subtle grain
        noise = np.random.normal(0, 7, out_arr.shape)
        out_arr = out_arr + noise

        out_arr = np.clip(out_arr, 0, 255).astype(np.uint8)

        out = Image.fromarray(out_arr)
        out = ImageEnhance.Contrast(out).enhance(1.22)
        out = ImageEnhance.Color(out).enhance(1.15)

        return out


    if v == "shape_cutout_blur_background":

        im = im.convert("RGB")
        w, h = im.size

        # Pick a random shape type
        shape = random.choice(["rectangle", "triangle", "circle"])

        # Pick random source area size
        src_w = random.randint(max(30, w // 5), max(60, w // 2))
        src_h = random.randint(max(30, h // 5), max(60, h // 2))

        sx = random.randint(0, max(0, w - src_w))
        sy = random.randint(0, max(0, h - src_h))

        source = im.crop((sx, sy, sx + src_w, sy + src_h))

        # Create mask for selected shape
        mask = Image.new("L", (src_w, src_h), 0)
        draw = ImageDraw.Draw(mask)

        if shape == "rectangle":
            draw.rectangle([0, 0, src_w, src_h], fill=255)

        elif shape == "circle":
            draw.ellipse([0, 0, src_w, src_h], fill=255)

        elif shape == "triangle":
            points = [
                (src_w // 2, 0),
                (0, src_h),
                (src_w, src_h)
            ]
            draw.polygon(points, fill=255)

        # Create a large blurry background from the selected piece
        bg = source.resize((w, h), Image.Resampling.BICUBIC)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=random.randint(18, 40)))

        bg = ImageEnhance.Brightness(bg).enhance(0.9)
        bg = ImageEnhance.Contrast(bg).enhance(1.15)

        out = bg.convert("RGBA")

        # Shrink selected piece
        scale = random.uniform(0.35, 0.65)
        small_w = max(10, int(src_w * scale))
        small_h = max(10, int(src_h * scale))

        small_piece = source.resize((small_w, small_h), Image.Resampling.BICUBIC)
        small_mask = mask.resize((small_w, small_h), Image.Resampling.BICUBIC)

        # Pick random destination for the small cutout
        dx = random.randint(0, max(0, w - small_w))
        dy = random.randint(0, max(0, h - small_h))

        # Add slight shadow behind the cutout
        shadow = Image.new("RGBA", (small_w, small_h), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)

        shadow_draw.bitmap((0, 0), small_mask, fill=(0, 0, 0, 120))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=6))

        out.alpha_composite(shadow, (dx + 6, dy + 6))

        # Paste the small shaped piece
        small_rgba = small_piece.convert("RGBA")
        small_rgba.putalpha(small_mask)

        out.alpha_composite(small_rgba, (dx, dy))

        return out.convert("RGB")

    if v == "cutout_orange_noir_background":

        im = im.convert("RGB")
        w, h = im.size

        # Pick the main cutout area
        cut_w = random.randint(max(30, w // 5), max(60, w // 2))
        cut_h = random.randint(max(30, h // 5), max(60, h // 2))

        cx = random.randint(0, max(0, w - cut_w))
        cy = random.randint(0, max(0, h - cut_h))

        cutout = im.crop((cx, cy, cx + cut_w, cy + cut_h))

        # Pick a second area that does not overlap too much with the cutout
        for _ in range(50):
            bg_w = random.randint(max(30, w // 5), max(60, w // 2))
            bg_h = random.randint(max(30, h // 5), max(60, h // 2))

            bx = random.randint(0, max(0, w - bg_w))
            by = random.randint(0, max(0, h - bg_h))

            overlap_x = max(0, min(cx + cut_w, bx + bg_w) - max(cx, bx))
            overlap_y = max(0, min(cy + cut_h, by + bg_h) - max(cy, by))
            overlap_area = overlap_x * overlap_y

            cut_area = cut_w * cut_h
            bg_area = bg_w * bg_h

            if overlap_area < min(cut_area, bg_area) * 0.25:
                break

        bg_piece = im.crop((bx, by, bx + bg_w, by + bg_h))

        # Enlarge second selected piece to full image size
        bg = bg_piece.resize((w, h), Image.Resampling.BICUBIC)

        # Apply orange noir jazz color grading to background
        gray = ImageOps.grayscale(bg)
        gray = ImageOps.autocontrast(gray)
        gray = ImageEnhance.Contrast(gray).enhance(1.75)
        gray = ImageEnhance.Brightness(gray).enhance(0.82)

        orange_noir = ImageOps.colorize(
            gray,
            black=(8, 5, 4),
            mid=(92, 42, 18),
            white=(245, 120, 22)
        )

        glow = orange_noir.filter(ImageFilter.GaussianBlur(radius=8))
        orange = Image.new("RGB", (w, h), (255, 105, 12))
        glow = Image.blend(glow, orange, 0.22)
        orange_noir = Image.blend(orange_noir, glow, 0.18)

        # Add slight blur to make the background feel enlarged and atmospheric
        orange_noir = orange_noir.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 4.5)))

        out = orange_noir.convert("RGBA")

        # Shrink the original cutout
        scale = random.uniform(0.35, 0.65)
        small_w = max(10, int(cut_w * scale))
        small_h = max(10, int(cut_h * scale))

        small_cutout = cutout.resize((small_w, small_h), Image.Resampling.BICUBIC)

        # Randomly place the small cutout
        dx = random.randint(0, max(0, w - small_w))
        dy = random.randint(0, max(0, h - small_h))

        # Add soft shadow
        shadow = Image.new("RGBA", (small_w, small_h), (0, 0, 0, 120))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=8))
        out.alpha_composite(shadow, (dx + 7, dy + 7))

        out.alpha_composite(small_cutout.convert("RGBA"), (dx, dy))

        return out.convert("RGB")



    if v == "transparent_checker_grid_pieces":

        im = im.convert("RGB")
        w, h = im.size

        out = im.convert("RGBA")

        # Create transparent checkerboard grid layer
        grid_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(grid_layer, "RGBA")

        # Random grid size
        cells_x = random.randint(6, 12)
        cells_y = random.randint(6, 12)

        cell_w = w / cells_x
        cell_h = h / cells_y

        # Transparent checkerboard colors
        color_a = (255, 255, 255, 55)
        color_b = (0, 0, 0, 55)

        for row in range(cells_y):
            for col in range(cells_x):

                x1 = int(col * cell_w)
                y1 = int(row * cell_h)
                x2 = int((col + 1) * cell_w)
                y2 = int((row + 1) * cell_h)

                if (row + col) % 2 == 0:
                    fill = color_a
                else:
                    fill = color_b

                draw.rectangle(
                    [x1, y1, x2, y2],
                    fill=fill,
                    outline=(255, 255, 255, 45)
                )

        # Pick two different random cells
        all_cells = [(row, col) for row in range(cells_y) for col in range(cells_x)]
        white_cell, black_cell = random.sample(all_cells, 2)

        def draw_piece(cell, color):
            row, col = cell

            x1 = int(col * cell_w)
            y1 = int(row * cell_h)
            x2 = int((col + 1) * cell_w)
            y2 = int((row + 1) * cell_h)

            # Circle size relative to cell
            margin = int(min(x2 - x1, y2 - y1) * 0.18)

            draw.ellipse(
                [x1 + margin, y1 + margin, x2 - margin, y2 - margin],
                fill=color,
                outline=(255, 255, 255, 160),
                width=max(2, int(min(x2 - x1, y2 - y1) * 0.04))
            )

        # Draw one white and one black circular piece in different cells
        draw_piece(white_cell, (255, 255, 255, 210))
        draw_piece(black_cell, (0, 0, 0, 220))

        out = Image.alpha_composite(out, grid_layer)

        return out.convert("RGB")



    if v == "smiley_overlay":

        im = im.convert("RGB")
        w, h = im.size

        out = im.convert("RGBA")

        # Create transparent layer for smiley faces
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")

        # Draw many simple smiley symbols
        count = random.randint(60, 160)

        for _ in range(count):

            size = random.randint(
                max(18, min(w, h) // 35),
                max(35, min(w, h) // 9)
            )

            x = random.randint(-size, w)
            y = random.randint(-size, h)

            alpha = random.randint(45, 135)

            color = random.choice([
                (255, 255, 255, alpha),
                (255, 230, 80, alpha),
                (0, 0, 0, alpha),
                (255, 40, 40, alpha)
            ])

            eye_r = max(2, size // 10)

            # Eye positions
            left_eye = (x + int(size * 0.35), y + int(size * 0.38))
            right_eye = (x + int(size * 0.65), y + int(size * 0.38))

            draw.ellipse(
                [left_eye[0] - eye_r, left_eye[1] - eye_r,
                 left_eye[0] + eye_r, left_eye[1] + eye_r],
                fill=color
            )

            draw.ellipse(
                [right_eye[0] - eye_r, right_eye[1] - eye_r,
                 right_eye[0] + eye_r, right_eye[1] + eye_r],
                fill=color
            )

            # Curved smile mouth
            mouth_box = [
                x + int(size * 0.25),
                y + int(size * 0.42),
                x + int(size * 0.75),
                y + int(size * 0.85)
            ]

            draw.arc(
                mouth_box,
                start=15,
                end=165,
                fill=color,
                width=max(2, size // 14)
            )

        # Slight blur so symbols feel embedded into the image
        layer = layer.filter(ImageFilter.GaussianBlur(radius=0.4))

        out = Image.alpha_composite(out, layer)

        return out.convert("RGB")



    if v == "circular_ripple":

        im = im.convert("RGB")
        w, h = im.size

        arr = np.array(im)
        out = arr.copy()

        # Pick a random circular ripple area
        cx = random.randint(int(w * 0.25), int(w * 0.75))
        cy = random.randint(int(h * 0.25), int(h * 0.75))

        radius = random.randint(
            max(40, min(w, h) // 5),
            max(80, min(w, h) // 3)
        )

        # Ripple settings
        amplitude = random.uniform(8, 22)
        wavelength = random.uniform(18, 38)

        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        dx = xx - cx
        dy = yy - cy
        dist = np.sqrt(dx * dx + dy * dy)

        # Circular mask
        mask = dist < radius

        # Ripple displacement, strongest near the center and fading outward
        fade = 1 - (dist / radius)
        fade = np.clip(fade, 0, 1)

        ripple = np.sin(dist / wavelength * 2 * math.pi) * amplitude * fade

        # Avoid division by zero in the center
        safe_dist = np.where(dist == 0, 1, dist)

        map_x = xx + (dx / safe_dist) * ripple
        map_y = yy + (dy / safe_dist) * ripple

        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)

        # Bilinear interpolation
        x0 = np.floor(map_x).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)

        y0 = np.floor(map_y).astype(np.int32)
        y1 = np.clip(y0 + 1, 0, h - 1)

        wx = map_x - x0
        wy = map_y - y0

        warped = (
            (1 - wx)[..., None] * (1 - wy)[..., None] * arr[y0, x0] +
            wx[..., None] * (1 - wy)[..., None] * arr[y0, x1] +
            (1 - wx)[..., None] * wy[..., None] * arr[y1, x0] +
            wx[..., None] * wy[..., None] * arr[y1, x1]
        )

        warped = np.clip(warped, 0, 255).astype(np.uint8)

        # Apply distortion only inside the circle
        out[mask] = warped[mask]

        # Add subtle bright/dark circular rings for water-ripple illusion
        ring_layer = Image.fromarray(out).convert("RGBA")
        draw = ImageDraw.Draw(ring_layer, "RGBA")

        for r in range(12, radius, int(wavelength)):
            alpha = int(90 * (1 - r / radius))
            width_s = random.randint(1, 3)

            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                outline=(255, 255, 255, alpha),
                width=width_s
            )

            if r + 4 < radius:
                draw.ellipse(
                    [cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4],
                    outline=(0, 0, 0, alpha // 3),
                    width=1
                )

        return ring_layer.convert("RGB")



    if v == "stylized_imagenet":

        im = im.convert("RGB")
        w, h = im.size

        # Preserve image content with softened structure
        base = im.filter(ImageFilter.SMOOTH_MORE)
        base = base.filter(ImageFilter.GaussianBlur(radius=0.7))

        # Strong color stylization
        base = ImageEnhance.Color(base).enhance(1.7)
        base = ImageEnhance.Contrast(base).enhance(1.25)
        base = ImageOps.posterize(base, bits=random.choice([4, 5]))

        arr = np.array(base).astype(np.float32)

        # Create synthetic neural-style texture noise
        noise = np.random.normal(0, 1, (h, w, 3)).astype(np.float32)

        noise_img = Image.fromarray(
            np.clip((noise - noise.min()) / (noise.max() - noise.min()) * 255, 0, 255).astype(np.uint8)
        )

        noise_img = noise_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 5.0)))
        noise_arr = np.array(noise_img).astype(np.float32)

        # Create brush-like texture using sinusoidal patterns
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        pattern = (
            np.sin(xx * random.uniform(0.025, 0.055) + yy * random.uniform(0.015, 0.045)) +
            np.sin(xx * random.uniform(0.045, 0.085) - yy * random.uniform(0.025, 0.065))
        )

        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        pattern = pattern[:, :, None]

        # Mix original colors with stylized texture
        styled = arr * 0.72 + noise_arr * 0.18 + pattern * 255 * 0.10

        styled = np.clip(styled, 0, 255).astype(np.uint8)
        out = Image.fromarray(styled)

        # Add edge-aware structure so the image remains readable
        gray = ImageOps.grayscale(im)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.autocontrast(edges)
        edges = edges.filter(ImageFilter.MaxFilter(3))

        edge_arr = np.array(edges)
        edge_mask = edge_arr > 45

        outline = Image.fromarray(
            np.where(edge_mask, 45, 255).astype(np.uint8)
        ).convert("RGB")

        out = ImageChops.multiply(out, outline)

        # Add painterly texture strokes
        draw = ImageDraw.Draw(out, "RGBA")
        arr2 = np.array(out).astype(np.int16)

        stroke_count = max(600, min(5000, (w * h) // 350))

        for _ in range(stroke_count):

            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)

            color = arr2[y, x]

            color = (
                int(np.clip(color[0] + random.randint(-25, 25), 0, 255)),
                int(np.clip(color[1] + random.randint(-25, 25), 0, 255)),
                int(np.clip(color[2] + random.randint(-25, 25), 0, 255)),
                random.randint(35, 85)
            )

            length = random.randint(5, 18)
            width_s = random.randint(1, 3)

            angle = (
                math.sin(y * 0.035) +
                math.cos(x * 0.025) +
                random.uniform(-1.0, 1.0)
            )

            x2 = x + int(math.cos(angle) * length)
            y2 = y + int(math.sin(angle) * length)

            draw.line(
                [(x, y), (x2, y2)],
                fill=color,
                width=width_s
            )

        # Final stylized correction
        out = ImageEnhance.Color(out).enhance(1.25)
        out = ImageEnhance.Contrast(out).enhance(1.15)
        out = ImageEnhance.Sharpness(out).enhance(1.35)

        return out

    if v == "classic_disney":

        im = im.convert("RGB")

        # Keep the original image readable
        base = im.filter(ImageFilter.SMOOTH_MORE)
        base = base.filter(ImageFilter.GaussianBlur(radius=0.6))

        # Soft warm color palette
        base = ImageEnhance.Color(base).enhance(1.45)
        base = ImageEnhance.Contrast(base).enhance(1.18)
        base = ImageEnhance.Brightness(base).enhance(1.08)

        # Reduce colors gently for hand-painted animation look
        base = ImageOps.posterize(base, bits=5)

        # Create clean cartoon outlines
        gray = ImageOps.grayscale(im)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.autocontrast(edges)
        edges = edges.filter(ImageFilter.MaxFilter(3))

        edge_arr = np.array(edges)
        edge_mask = edge_arr > 45

        outline = Image.fromarray(
            np.where(edge_mask, 35, 255).astype(np.uint8)
        ).convert("RGB")

        out = ImageChops.multiply(base, outline)

        # Add soft golden animation warmth
        warm = Image.new("RGB", out.size, (255, 225, 185))
        out = Image.blend(out, warm, 0.10)

        # Add subtle dreamy glow
        glow = out.filter(ImageFilter.GaussianBlur(radius=4))
        out = Image.blend(out, glow, 0.16)

        # Final sharpening for animated-cell clarity
        out = ImageEnhance.Sharpness(out).enhance(1.45)

        return out


    if v == "random_piece_swap":

        im = im.convert("RGB")
        w, h = im.size

        out = im.copy()

        # Pick random piece size
        piece_w = random.randint(max(20, w // 8), max(40, w // 3))
        piece_h = random.randint(max(20, h // 8), max(40, h // 3))

        # Pick first source position
        x1 = random.randint(0, max(0, w - piece_w))
        y1 = random.randint(0, max(0, h - piece_h))

        # Pick second destination position
        x2 = random.randint(0, max(0, w - piece_w))
        y2 = random.randint(0, max(0, h - piece_h))

        # Crop both pieces before pasting
        piece_a = im.crop((x1, y1, x1 + piece_w, y1 + piece_h))
        piece_b = im.crop((x2, y2, x2 + piece_w, y2 + piece_h))

        # Swap their positions
        out.paste(piece_a, (x2, y2))
        out.paste(piece_b, (x1, y1))

        return out
    

    if v == "double_piece_swap_purple_spatter":

        im = im.convert("RGB")
        w, h = im.size

        out = im.copy()

        changed_regions = []

        # Apply two independent random piece swaps
        for _ in range(2):

            piece_w = random.randint(max(20, w // 10), max(40, w // 4))
            piece_h = random.randint(max(20, h // 10), max(40, h // 4))

            x1 = random.randint(0, max(0, w - piece_w))
            y1 = random.randint(0, max(0, h - piece_h))

            x2 = random.randint(0, max(0, w - piece_w))
            y2 = random.randint(0, max(0, h - piece_h))

            piece_a = out.crop((x1, y1, x1 + piece_w, y1 + piece_h))
            piece_b = out.crop((x2, y2, x2 + piece_w, y2 + piece_h))

            out.paste(piece_a, (x2, y2))
            out.paste(piece_b, (x1, y1))

            changed_regions.append((x1, y1, piece_w, piece_h))
            changed_regions.append((x2, y2, piece_w, piece_h))

        out = out.convert("RGBA")

        # Tint swapped regions purple
        purple_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        purple_draw = ImageDraw.Draw(purple_layer, "RGBA")

        for x, y, pw, ph in changed_regions:
            purple_draw.rectangle(
                [x, y, x + pw, y + ph],
                fill=(125, 45, 180, 95)
            )

        out = Image.alpha_composite(out, purple_layer)

        # Add dark purple spatter only over changed regions
        spatter_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        spatter_draw = ImageDraw.Draw(spatter_layer, "RGBA")

        for x, y, pw, ph in changed_regions:

            # Dense splatter inside each modified piece
            for _ in range(random.randint(35, 90)):

                cx = random.randint(x, min(w - 1, x + pw))
                cy = random.randint(y, min(h - 1, y + ph))

                base_r = random.randint(2, max(3, min(pw, ph) // 12))

                color = random.choice([
                    (45, 0, 80, random.randint(90, 170)),
                    (70, 0, 115, random.randint(80, 155)),
                    (95, 20, 140, random.randint(70, 145))
                ])

                # Build each spatter blob from smaller circles
                for _ in range(random.randint(4, 12)):

                    angle = random.uniform(0, 2 * math.pi)
                    dist = random.uniform(0, base_r * 1.8)

                    px = cx + int(math.cos(angle) * dist)
                    py = cy + int(math.sin(angle) * dist)

                    if x <= px <= x + pw and y <= py <= y + ph:
                        r = random.randint(1, max(2, base_r // 2))

                        spatter_draw.ellipse(
                            [px - r, py - r, px + r, py + r],
                            fill=color
                        )

        spatter_layer = spatter_layer.filter(ImageFilter.GaussianBlur(radius=0.45))

        out = Image.alpha_composite(out, spatter_layer)

        return out.convert("RGB")

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
