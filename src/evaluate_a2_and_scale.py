"""Evaluasi Eksperimen A2 (Regional Weighted Fit) & Skala Fisik (sigma_mm & feather_mm).

Menguji:
1. Apakah A2 (bobot rendah 0.25 pada cincin tepi) menurunkan galat di area inti wajah (hidung, mata, bibir)?
2. Apakah nRMSE keseluruhan membaik jika titik-titik tepi rahang tidak lagi mendistorsi fit?
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import relief_core as rc
import relief_improve as ri


def run_a2_evaluation(cache_dir: str = "cache", out_dir: str = "experiments"):
    c_path = Path(cache_dir)
    exp_path = Path(out_dir)
    meta_files = sorted(list(c_path.glob("*_meta.json")))

    if not meta_files:
        print(f"Tidak ada cache ditemukan di {c_path}")
        return

    rows = []
    print(f"[*] Mengevaluasi {len(meta_files)} foto dari cache...")

    for m_file in meta_files:
        stem = m_file.name.replace("_meta.json", "")
        lm_file = c_path / f"{stem}_landmarks.npy"
        depth_file = c_path / f"{stem}_depth.npy"

        with open(m_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        lm = np.load(lm_file)
        depth = np.load(depth_file)
        W, H = meta["W"], meta["H"]
        mmpp = meta["mm_per_px"]

        pts, h_lm = rc.landmarks_to_px_and_height(lm, W, H, mmpp)
        d_at = rc.sample_map(depth, pts)

        # 1. Baseline Affine
        fit_base = rc.fit_affine_robust(d_at, h_lm)
        nrmse_base = ri.normalized_rmse(fit_base["rmse_mm"], h_lm)
        resid_base = h_lm - (fit_base["a"] * d_at + fit_base["b"])

        nose_pt = pts[1]  # Ujung hidung
        rings_base = ri.residuals_by_ring(pts, resid_base, nose_pt)
        core_base = rings_base["inti"][1]

        # 2. A2: Regional Weighted Affine (1.0 inti, 0.8 tengah, 0.25 tepi)
        prior_w = ri.regional_weights(pts, nose_pt, edges=(0.4, 0.75), weights=(1.0, 0.8, 0.25))
        fit_a2 = ri.fit_affine_weighted_robust(d_at, h_lm, prior_weights=prior_w)
        nrmse_a2 = ri.normalized_rmse(fit_a2["rmse_mm"], h_lm)
        resid_a2 = h_lm - (fit_a2["a"] * d_at + fit_a2["b"])
        rings_a2 = ri.residuals_by_ring(pts, resid_a2, nose_pt)
        core_a2 = rings_a2["inti"][1]

        # Konversi skala mm ke piksel
        sigma_px = ri.sigma_mm_to_px(3.0, mmpp)
        feather_px = ri.sigma_mm_to_px(5.0, mmpp)

        # Penurunan galat inti
        core_improvement_pct = ((core_base - core_a2) / core_base) * 100.0

        rows.append({
            "foto": stem[:25] + "...",
            "mm_per_px": round(mmpp, 3),
            "sigma_px_adaptif": round(sigma_px, 1),
            "core_rmse_base": round(core_base, 2),
            "core_rmse_A2": round(core_a2, 2),
            "delta_core_pct": round(core_improvement_pct, 1),
            "nrmse_base": round(nrmse_base, 3),
            "nrmse_A2": round(nrmse_a2, 3),
            "a_base": round(fit_base["a"], 1),
            "a_A2": round(fit_a2["a"], 1),
            "core_improved": core_a2 < core_base,
        })

    df = pd.DataFrame(rows)
    csv_file = exp_path / "A2_weighted_fit_comparison.csv"
    df.to_csv(csv_file, index=False)

    print("\n" + "=" * 105)
    print(" HASIL EVALUASI A2: REGIONAL WEIGHTED FIT (BOBOT PRIOR INTI 1.0 vs TEPI 0.25)")
    print("=" * 105)
    print(df.to_string(index=False))
    print("=" * 105)

    n_total = len(df)
    n_improved = df["core_improved"].sum()
    avg_core_base = df["core_rmse_base"].mean()
    avg_core_a2 = df["core_rmse_A2"].mean()
    avg_nrmse_base = df["nrmse_base"].mean()
    avg_nrmse_a2 = df["nrmse_A2"].mean()

    print(f"\n[RINGKASAN HASIL A2]:")
    print(f" - Galat Inti Wajah Membaik pada : {n_improved}/{n_total} foto ({n_improved/n_total*100:.1f}%)")
    print(f" - Rata-rata Galat Inti Baseline : {avg_core_base:.2f} mm")
    print(f" - Rata-rata Galat Inti A2       : {avg_core_a2:.2f} mm ({((avg_core_base - avg_core_a2)/avg_core_base)*100:.1f}% perbaikan)")
    print(f" - Rata-rata nRMSE Keseluruhan   : {avg_nrmse_base:.4f} -> {avg_nrmse_a2:.4f}")
    print(f" - Data CSV lengkap disimpan di  : {csv_file}")


if __name__ == "__main__":
    run_a2_evaluation()
