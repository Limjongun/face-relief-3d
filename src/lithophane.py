"""Generator Lithophane 3D-Printable dari Foto 2D.

Lithophane adalah plakat tipis di mana area gelap dibuat lebih tebal
dan area terang dibuat lebih tipis, sehingga gambar foto tampak saat disinari lampu dari belakang.

Cara Pakai CLI:
python src/lithophane.py path/ke/foto.jpg --out out/lithophane --width-mm 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import relief_core as rc
except ImportError:
    from src import relief_core as rc


def create_lithophane(
    image_path: str,
    out_dir: str = "out/lithophane",
    width_mm: float = 100.0,
    max_grid_px: int = 500,
    t_min_mm: float = 0.8,
    t_max_mm: float = 3.0,
    gamma: float = 1.0,
    border_mm: float = 3.0,
    border_thick_mm: float = 3.5,
) -> dict:
    """Mengonversi foto 2D menjadi berkas STL lithophane dengan frame pembatas."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Baca citra dan orientasi EXIF
    pil_img = ImageOps.exif_transpose(Image.open(image_path)).convert("L")  # grayscale
    gray_np = np.array(pil_img, dtype=np.uint8)

    # 2. Terapkan CLAHE (Contrast Limited Adaptive Histogram Equalization) untuk memperjelas transisi abu-abu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_np)

    # 3. Resize dengan aspect ratio yang sama agar jumlah segitiga pas
    orig_h, orig_w = enhanced.shape
    scale = min(1.0, max_grid_px / orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Skala fisik dalam mm
    pixel_mm = width_mm / new_w
    height_mm = new_h * pixel_mm

    # 4. Hitung ketebalan per piksel
    # Formula: terang (1.0) -> tipis (t_min); gelap (0.0) -> tebal (t_max)
    gray01 = resized.astype(np.float64) / 255.0
    thickness = rc.lithophane_thickness(gray01, t_min_mm=t_min_mm, t_max_mm=t_max_mm, gamma=gamma)

    # Tambahkan bingkai / border opsional di sekeliling lithophane agar kokoh saat dipajang
    if border_mm > 0:
        b_px = int(np.round(border_mm / pixel_mm))
        if b_px > 0:
            thickness[:b_px, :] = border_thick_mm
            thickness[-b_px:, :] = border_thick_mm
            thickness[:, :b_px] = border_thick_mm
            thickness[:, -b_px:] = border_thick_mm

    # 5. Konversi ke mesh padat watertight
    # Di relief_core: z_top = base_mm + h_mm. Di sini base_mm = t_min_mm, h_mm = thickness - t_min_mm
    h_relief = np.maximum(0.0, thickness - t_min_mm)
    tris = rc.heightmap_to_solid(h_relief, pixel_mm=pixel_mm, base_mm=t_min_mm)

    info = rc.check_mesh(tris)
    stl_file = out_path / "lithophane.stl"
    rc.write_binary_stl(str(stl_file), tris, header="face-lithophane")

    report = {
        "image_path": str(image_path),
        "dimensions_mm": {
            "width": float(new_w * pixel_mm),
            "height": float(new_h * pixel_mm),
            "t_min": t_min_mm,
            "t_max": t_max_mm,
        },
        "grid_size": [new_h, new_w],
        "pixel_mm": float(pixel_mm),
        "mesh": info,
    }

    with open(out_path / "lithophane_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[✓] Lithophane STL berhasil dibuat: {stl_file}")
    print(f"    Ukuran: {report['dimensions_mm']['width']:.1f} x {report['dimensions_mm']['height']:.1f} mm")
    print(f"    Ketebalan: {t_min_mm:.1f} mm s/d {t_max_mm:.1f} mm")
    print(f"    Watertight: {info['watertight']}")
    return report


def main():
    p = argparse.ArgumentParser(description="Buat STL Lithophane dari Foto Wajah 2D")
    p.add_argument("image", help="Path ke file foto")
    p.add_argument("--out", default="out/lithophane", help="Folder keluaran")
    p.add_argument("--width-mm", type=float, default=100.0, help="Lebar fisik hasil cetak (mm)")
    p.add_argument("--max-px", type=int, default=400, help="Resolusi maksimum sisi terpanjang grid")
    p.add_argument("--t-min", type=float, default=0.8, help="Ketebalan minimum pada area paling putih (mm)")
    p.add_argument("--t-max", type=float, default=3.0, help="Ketebalan maksimum pada area paling gelap (mm)")
    p.add_argument("--gamma", type=float, default=1.0, help="Koreksi gamma kontras ketebalan")
    p.add_argument("--border-mm", type=float, default=2.5, help="Lebar bingkai penguat (mm)")
    p.add_argument("--border-thick-mm", type=float, default=3.2, help="Ketebalan bingkai penguat (mm)")
    args = p.parse_args()

    create_lithophane(
        image_path=args.image,
        out_dir=args.out,
        width_mm=args.width_mm,
        max_grid_px=args.max_px,
        t_min_mm=args.t_min,
        t_max_mm=args.t_max,
        gamma=args.gamma,
        border_mm=args.border_mm,
        border_thick_mm=args.border_thick_mm,
    )


if __name__ == "__main__":
    main()
