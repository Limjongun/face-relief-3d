"""Membuat tabel ringkasan metrik kualitas untuk out/batch_deep.

Menghitung/mengambil:
- fit.a: polaritas/kemiringan affine (> 0 = valid)
- fit.rmse_mm: residual kecocokan landmark vs depth
- fit.inlier_frac: fraksi konsistensi (inlier)
- mm_per_px: skala metrik dari IPD iris
- volume: volume mesh dalam mm³
- flag: True jika a <= 0 atau inlier < 0.6 (kandidat gagal)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import relief_core as rc
from run_relief import get_landmarks


def generate_summary(batch_dir: str = "out/batch_deep", input_dir: str = "data/my_photos"):
    batch_path = Path(batch_dir)
    in_path = Path(input_dir)

    report_files = sorted(list(batch_path.glob("*/report.json")))
    if not report_files:
        print(f"Tidak ada file report.json ditemukan di {batch_path}")
        return

    # Muat depth model sekali saja jika ada report yang belum punya data fit
    pipe = None
    model_path = "models/face_landmarker.task"

    rows = []
    print(f"[*] Menganalisis metrik kualitas dari {len(report_files)} folder...")

    for rep_file in report_files:
        folder_dir = rep_file.parent
        folder_name = folder_dir.name
        with open(rep_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        fit = data.get("fit", {})
        mesh = data.get("mesh", {})
        mmpp = data.get("mm_per_px", None)

        # Jika fit belum tersimpan di report.json lama, hitung dari citra asli
        if not fit or "a" not in fit:
            img_name = data.get("file", "")
            img_file = in_path / img_name
            if img_file.exists():
                try:
                    if pipe is None:
                        from transformers import pipeline
                        print("[*] Memuat pipeline depth pada CPU...")
                        pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)

                    img = ImageOps.exif_transpose(Image.open(img_file)).convert("RGB")
                    rgb = np.array(img)
                    H, W = rgb.shape[:2]

                    lm = get_landmarks(rgb, model_path)
                    iris_a = lm[rc.IRIS_CENTER_A, :2] * [W, H]
                    iris_b = lm[rc.IRIS_CENTER_B, :2] * [W, H]
                    mmpp = rc.mm_per_pixel_from_ipd(iris_a, iris_b, 63.0)
                    pts, h_lm = rc.landmarks_to_px_and_height(lm, W, H, mmpp)

                    out_p = pipe(img)
                    pred = out_p.get("predicted_depth", None)
                    if hasattr(pred, "squeeze"): pred = pred.squeeze()
                    if hasattr(pred, "cpu"): pred = pred.cpu()
                    if hasattr(pred, "numpy"): d = pred.numpy().astype(np.float32)
                    else: d = np.asarray(pred, dtype=np.float32)
                    depth = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)

                    fit = rc.fit_affine_robust(rc.sample_map(depth, pts), h_lm)

                    # Update file report.json agar permanen
                    data["fit"] = fit
                    data["mm_per_px"] = round(float(mmpp), 4)
                    with open(rep_file, "w", encoding="utf-8") as f_up:
                        json.dump(data, f_up, indent=2)
                except Exception as e:
                    print(f"    Gagal mengekstrak ulang {img_name}: {e}")

        # Ambil volume mesh jika ada di mesh info atau hitung dari bbox
        vol = mesh.get("volume_mm3", 0.0)
        if vol == 0.0 and "volume" in data:
            vol = data.get("volume", 0.0)

        a_val = fit.get("a", 0.0)
        rmse_val = fit.get("rmse_mm", 0.0)
        inl_val = fit.get("inlier_frac", 0.0)

        rows.append({
            "folder": folder_name,
            "a": round(float(a_val), 3),
            "rmse_mm": round(float(rmse_val), 3),
            "inlier": round(float(inl_val), 3),
            "mm_per_px": round(float(mmpp or 0.0), 3),
            "volume": round(float(vol), 1),
            "watertight": mesh.get("watertight", data.get("watertight", True)),
        })

    df = pd.DataFrame(rows).sort_values("folder")
    df["flag"] = (df["a"] <= 0) | (df["inlier"] < 0.6)

    csv_out = batch_path / "quality_summary.csv"
    df.to_csv(csv_out, index=False)

    print("\n" + "=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)
    print(f"\n[✓] Ringkasan metrik kualitas tersimpan di: {csv_out}")
    return df


if __name__ == "__main__":
    generate_summary()
