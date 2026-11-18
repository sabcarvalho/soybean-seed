"""Gera .bil + .hdr mockados (ENVI) a partir de um RGB mock.

As posições das sementes são detectadas no RGB e uma assinatura
espectral distinta é pintada em cada blob (uma gaussiana em uma banda
diferente por classe), para que a classificação tenha sinal real.

Uso:
    # uma imagem
    python scripts/generate_mock_hyper.py \
        --rgb data/labeled/RGB/RIP2_001-RGB.tif \
        --out-dir data/labeled \
        --id 001 \
        --n-classes 4

    # pasta inteira
    python scripts/generate_mock_hyper.py \
        --rgb-dir data/labeled/RGB \
        --out-dir data/labeled \
        --n-classes 4
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np
from spectral import envi


# ---------------------------------------------------------------- writers

def write_envi_header(path, samples, lines, bands, data_type, interleave,
                      byte_order, description="", wavelength=None):
    header = f"""ENVI
description = {{{description}}}
samples = {samples}
lines = {lines}
bands = {bands}
header offset = 0
file type = ENVI Standard
data type = {data_type}
interleave = {interleave}
byte order = {byte_order}
"""
    if wavelength is not None:
        header += "wavelength units = Nanometers\n"
        header += "wavelength = {\n"
        header += ",\n".join(f"{w:.2f}" for w in wavelength)
        header += "\n}\n"
    with open(path, "w") as f:
        f.write(header)


def write_bil(path, cube):
    """(bands, lines, samples) -> BIL em disco."""
    # BIL = para cada linha, todos os bandas, todas as amostras
    cube.transpose(1, 0, 2).astype(np.float32).tofile(path)


def write_bsq(path, cube):
    """(bands, lines, samples) -> BSQ em disco (mais simples)."""
    cube.astype(np.float32).tofile(path)


# ------------------------------------------------------------ blob detect

def detect_blobs(rgb_bgr, min_area=30):
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        blobs.append((x + w // 2, y + h // 2, w, h))
    return blobs


# -------------------------------------------------------------- generator

def make_signature(bands, class_id, n_classes):
    """Gaussiana centrada na banda `class_id/n_classes`."""
    x = np.linspace(0, 1, bands)
    peak = (class_id + 0.5) / n_classes
    return 0.1 + 0.9 * np.exp(-((x - peak) ** 2) / (2 * 0.15 ** 2))


def generate_cube(shape, blobs, n_bands=150, n_classes=4, seed=42,
                  bg=0.1, bg_noise=0.02):
    rng = np.random.default_rng(seed)
    H, W = shape
    cube = np.full((n_bands, H, W), bg, dtype=np.float32)
    cube += rng.normal(0, bg_noise, cube.shape).astype(np.float32)

    for i, (cx, cy, w, h) in enumerate(blobs):
        sig = make_signature(n_bands, i % n_classes, n_classes).astype(np.float32)
        x0, x1 = max(0, cx - w // 2), min(W, cx + w // 2)
        y0, y1 = max(0, cy - h // 2), min(H, cy + h // 2)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (((xx - cx) ** 2) / (max(w / 2, 1) ** 2) +
                ((yy - cy) ** 2) / (max(h / 2, 1) ** 2)) <= 1.0
        for b in range(n_bands):
            cube[b, y0:y1, x0:x1][mask] = sig[b]

    return cube


# ------------------------------------------------------------------- main

_ID_RE = re.compile(r"RIP2_(?P<id>.+?)-RGB\.\w+$")


def _process(rgb_path, out_dir, image_id, args):
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        print(f"[skip] não consegui ler {rgb_path}")
        return
    H, W = rgb.shape[:2]
    blobs = detect_blobs(rgb, min_area=args.min_area)

    cube = generate_cube((H, W), blobs, n_bands=args.bands,
                         n_classes=args.n_classes, seed=args.seed)

    bil_path = os.path.join(out_dir, f"RIP2_{image_id}.bil")
    hdr_path = os.path.join(out_dir, f"RIP2_{image_id}.bil.hdr")

    (write_bil if args.interleave == "bil" else write_bsq)(bil_path, cube)
    write_envi_header(
        hdr_path,
        samples=W, lines=H, bands=args.bands,
        data_type=4,                # float32
        interleave=args.interleave,
        byte_order=0,               # little-endian
        description="mock hyperspectral for pipeline test",
        wavelength=np.linspace(400, 1000, args.bands),
    )
    size_mb = os.path.getsize(bil_path) / 1e6
    print(f"[ok] {image_id}: {W}x{H}x{args.bands} "
          f"({len(blobs)} blobs, {args.interleave.upper()}, {size_mb:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rgb")
    p.add_argument("--rgb-dir")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--id", help="ID (obrigatório com --rgb)")
    p.add_argument("--bands", type=int, default=150)
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-area", type=int, default=30)
    p.add_argument("--interleave", choices=["bil", "bsq"], default="bsq")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.rgb:
        if not args.id:
            p.error("--id é obrigatório junto de --rgb")
        _process(args.rgb, args.out_dir, args.id, args)
        return

    if args.rgb_dir:
        for path in sorted(glob.glob(os.path.join(args.rgb_dir, "*.tif"))) + \
                    sorted(glob.glob(os.path.join(args.rgb_dir, "*.tiff"))) + \
                    sorted(glob.glob(os.path.join(args.rgb_dir, "*.png"))):
            m = _ID_RE.search(os.path.basename(path))
            if not m:
                continue
            _process(path, args.out_dir, m.group("id"), args)
        return

    p.error("informe --rgb ou --rgb-dir")


if __name__ == "__main__":
    main()