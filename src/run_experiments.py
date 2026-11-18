"""Skrip Eksperimen & Caching: Evaluasi Baseline vs Peningkatan Akurasi (A1: Affine vs Poly vs Disparitas).

Fitur:
1. Caching: Menyimpan landmark dan depth map ke folder cache/ sehingga iterasi eksperimen berjalan instan (< 0.1 detik per foto).
2. Menghitung metrik akurasi sejati: nRMSE (normalisasi rentang 5-95), residu per cincin (inti/tengah/tepi), simetri, dan konsistensi shading.
3. Menguji hipotesis A1: Apakah model nonlinear (polinomial atau disparitas) menurunkan nRMSE pada foto riil.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import relief_core as rc
import relief_improve as ri
from run_relief import get_landmarks


def cache_image_data(img_path: Path, cache_dir: Path, landmarker_path: str, pipe) -> dict:
    """Ekstraksi dan caching landmark, depth map, dan metadata foto."""
    stem = img_path.stem
    lm_cache = cache_dir / f"{stem}_landmarks.npy"
    depth_cache = cache_dir / f"{stem}_depth.npy"
    meta_cache = cache_dir / f"{stem}_meta.json"

    if lm_cache.exists() and depth_cache.exists() and meta_cache.exists():
        lm = np.load(lm_cache)
        depth = np.load(depth_cache)
        with open(meta_cache, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return {"lm": lm, "depth": depth, "meta": meta, "from_cache": True}

    img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    rgb = np.array(img)
    H, W = rgb.shape[:2]

    # Ekstraksi Landmark MediaPipe
    lm = get_landmarks(rgb, landmarker_path)

    # Ekstraksi Depth Anything V2
    out_pipe = pipe(img)
    pred = out_pipe.get("predicted_depth", None)
    if pred is None and "depth" in out_pipe:
        pred = np.array(out_pipe["depth"], dtype=np.float32)
    if hasattr(pred, "squeeze"): pred = pred.squeeze()
    if hasattr(pred, "cpu"): pred = pred.cpu()
    if hasattr(pred, "numpy"): d = pred.numpy().astype(np.float32)
    else: d = np.asarray(pred, dtype=np.float32)
    depth = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)

    # Hitung Skala Fisik IPD
    iris_a = lm[rc.IRIS_CENTER_A, :2] * [W, H]
    iris_b = lm[rc.IRIS_CENTER_B, :2] * [W, H]
    mmpp = rc.mm_per_pixel_from_ipd(iris_a, iris_b, 63.0)

    meta = {
        "W": W,
        "H": H,
        "mm_per_px": float(mmpp),
        "iris_a": iris_a.tolist(),
        "iris_b": iris_b.tolist(),
    }

    np.save(lm_cache, lm)
    np.save(depth_cache, depth)
    with open(meta_cache, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"lm": lm, "depth": depth, "meta": meta, "from_cache": False}


def run_evaluation(input_dir: str = "data/my_photos", cache_dir: str = "cache", out_dir: str = "experiments"):
    in_path = Path(input_dir)
    cache_path = Path(cache_dir)
    exp_path = Path(out_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    exp_path.mkdir(parents=True, exist_ok=True)

    img_files = sorted([f for f in in_path.glob("*.jpg")])
    if not img_files:
        print(f"Tidak ada foto ditemukan di {in_path}")
        return

    print("=" * 80)
    print(f" PROTOKOL EVALUASI AKURASI & BENCHMARK A1 (Total: {len(img_files)} foto)")
    print("=" * 80)

    pipe = None
    landmarker_path = "models/face_landmarker.task"

    # Periksa apakah ada yang perlu di-cache
    need_model = any(not (cache_path / f"{f.stem}_landmarks.npy").exists() for f in img_files)
    if need_model:
        from transformers import pipeline
        print("[*] Memuat model depth pada CPU untuk pembuatan cache pertama kali...")
        pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)

    cached_data = []
    print("[*] Memeriksa/Membangun cache per gambar...")
    t0_cache = time.time()
    for f in img_files:
        d = cache_image_data(f, cache_path, landmarker_path, pipe)
        cached_data.append((f, d))
    print(f"[✓] Semua cache siap dalam {time.time() - t0_cache:.2f} detik.\n")

    # Evaluasi Metrik
    rows = []
    print("[*] Mengevaluasi Affine vs Polinomial vs Disparitas...")

    for img_file, data in cached_data:
        lm = data["lm"]
        depth = data["depth"]
        meta = data["meta"]
        W, H = meta["W"], meta["H"]
        mmpp = meta["mm_per_px"]

        pts, h_lm = rc.landmarks_to_px_and_height(lm, W, H, mmpp)
        d_at = rc.sample_map(depth, pts)

        # 1. Baseline: Affine Alignment
        fit_aff = rc.fit_affine_robust(d_at, h_lm)
        nrmse_aff = ri.normalized_rmse(fit_aff["rmse_mm"], h_lm)
        resid_aff = h_lm - (fit_aff["a"] * d_at + fit_aff["b"])

        # Ujung hidung (landmark 1 atau 4 pada MediaPipe Face Mesh)
        nose_idx = 1
        nose_pt = pts[nose_idx]
        rings_aff = ri.residuals_by_ring(pts, resid_aff, nose_pt)

        # 2. A1 Varian: Fit Polinomial (deg=2)
        fit_poly = ri.fit_poly_robust(d_at, h_lm, deg=2)
        nrmse_poly = ri.normalized_rmse(fit_poly["rmse_mm"], h_lm)

        # 3. A1 Varian: Fit Model Disparitas (h = a/(d+s) + b)
        fit_dsp = ri.fit_disparity_model(d_at, h_lm)
        nrmse_dsp = ri.normalized_rmse(fit_dsp["rmse_mm"], h_lm)

        # Mask untuk simetri dan shading
        hull_pts = rc.convex_hull_polygon(pts)
        mask = rc.polygon_mask((H, W), hull_pts)

        # Relief cepat untuk simetri & shading
        aligned_aff = fit_aff["a"] * depth + fit_aff["b"]
        sym_err = ri.mirror_symmetry_error(aligned_aff, mask, axis_x=nose_pt[0])

        img_gray = ImageOps.exif_transpose(Image.open(img_file)).convert("L")
        gray01 = np.array(img_gray, dtype=float) / 255.0
        shd_corr = ri.shading_consistency(aligned_aff, mmpp, gray01, mask)

        rows.append({
            "file": img_file.name[:30] + "...",
            "mm_per_px": round(mmpp, 3),
            "rmse_aff": round(fit_aff["rmse_mm"], 2),
            "nrmse_aff": round(nrmse_aff, 3),
            "nrmse_poly": round(nrmse_poly, 3),
            "nrmse_dsp": round(nrmse_dsp, 3),
            "inti_mm": round(rings_aff["inti"][1], 2),
            "tepi_mm": round(rings_aff["tepi"][1], 2),
            "simetri_mm": round(sym_err, 2),
            "shading_r": round(shd_corr, 2),
            "best_model": "Poly" if nrmse_poly < min(nrmse_aff, nrmse_dsp) else ("Disparity" if nrmse_dsp < nrmse_aff else "Affine")
        })

    df = pd.DataFrame(rows)
    csv_out = exp_path / "A1_fit_comparison.csv"
    df.to_csv(csv_out, index=False)

    print("\n" + "=" * 105)
    print(" HASIL PERBANDINGAN METRIK AKURASI: AFFINE vs POLINOMIAL vs DISPARITAS")
    print("=" * 105)
    print(df.to_string(index=False))
    print("=" * 105)

    # Ringkasan perbandingan
    poly_better = (df["nrmse_poly"] < df["nrmse_aff"]).sum()
    dsp_better = (df["nrmse_dsp"] < df["nrmse_aff"]).sum()
    n_total = len(df)

    avg_nrmse_aff = df["nrmse_aff"].mean()
    avg_nrmse_poly = df["nrmse_poly"].mean()
    avg_nrmse_dsp = df["nrmse_dsp"].mean()

    summary_md = f"""# Ringkasan Eksperimen A1 — Penyelarasan Nonlinear

## 1. Rata-rata nRMSE (Normalized RMSE)
* **Baseline (Affine)**: `{avg_nrmse_aff:.4f}`
* **Varian 1 (Polinomial deg=2)**: `{avg_nrmse_poly:.4f}` ({'Lebih Baik' if avg_nrmse_poly < avg_nrmse_aff else 'Lebih Buruk'})
* **Varian 2 (Disparitas h=a/(d+s)+b)**: `{avg_nrmse_dsp:.4f}` ({'Lebih Baik' if avg_nrmse_dsp < avg_nrmse_aff else 'Lebih Buruk'})

## 2. Kriteria Penerimaan (Diterima jika nRMSE turun pada ≥ 80% foto)
* Polinomial mengalahkan Affine pada: **{poly_better}/{n_total} foto ({poly_better/n_total*100:.1f}%)**
* Model Disparitas mengalahkan Affine pada: **{dsp_better}/{n_total} foto ({dsp_better/n_total*100:.1f}%)**

## 3. Analisis Residu Cincin (Inti vs Tepi)
* Rata-rata galat inti hidung/mata: `{df['inti_mm'].mean():.2f} mm`
* Rata-rata galat tepi rahang: `{df['tepi_mm'].mean():.2f} mm`
"""
    with open(exp_path / "E1_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"\n[✓] Laporan evaluasi disimpan di: {exp_path / 'E1_summary.md'}")
    print(f"[✓] Data CSV tersimpan di: {csv_out}")


if __name__ == "__main__":
    run_evaluation()
