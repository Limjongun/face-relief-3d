"""Batch Processing: Konversi Banyak Foto Wajah Sekaligus Menjadi Relief 3D STL.

Fitur:
- Memuat model AI ke GPU RTX hanya 1 KALI (jauh lebih cepat, ~1-2 detik per foto).
- Menghasilkan file STL, preview 478 landmark wajah, colormap depth, dan laporan untuk tiap foto.
- Melanjutkan otomatis jika ada salah satu foto yang buram/gagal.

Cara Pakai CLI:
python src/batch_relief.py --input-dir data/my_photos --out-dir out/batch
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import relief_core as rc
    import relief_improve as ri
    from run_relief import get_landmarks, save_landmark_visualization
except ImportError:
    from src import relief_core as rc
    from src import relief_improve as ri
    from src.run_relief import get_landmarks, save_landmark_visualization


def sanitize_folder_name(name: str, index: int | None = None) -> str:
    """Membersihkan nama file panjang menjadi nama folder yang unik dan rapi.
    Menjaga bagian akhir nama file agar identitas unik (seperti aug_125) tidak hilang.
    """
    clean = re.sub(r"[^\w\-.]", "_", name)
    clean = re.sub(r"_+", "_", clean).strip("_")
    # Jika nama file sangat panjang, ambil 30 huruf depan + 35 huruf belakang (tempat nomor frame/aug)
    if len(clean) > 65:
        clean = f"{clean[:25]}_{clean[-35:]}"
    if index is not None:
        return f"{index:02d}_{clean}"
    return clean or "item"


def run_batch(args: argparse.Namespace):
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[X] Direktori masukan '{input_dir}' tidak ditemukan!")
        sys.exit(1)

    # Ekstensi gambar yang didukung
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]

    if not image_files:
        print(f"[!] Tidak ada file gambar yang ditemukan di '{input_dir}'")
        return

    print("=" * 60)
    print(f" BATCH RELIEF 3D PROCESSOR")
    print(f" Ditemukan: {len(image_files)} gambar di '{input_dir}'")
    print(f" Output   : '{out_dir}'")
    print("=" * 60)

    # 1. Muat Pipeline Depth Anything V2 SEKALI SAJA di GPU
    print("[*] Memuat model Depth Anything V2 ke memori GPU...")
    import torch
    from transformers import pipeline

    dev = 0 if torch.cuda.is_available() and args.device != "cpu" else -1
    device_name = torch.cuda.get_device_name(0) if dev == 0 else "CPU"
    print(f"[*] Menggunakan device: {device_name}")

    pipe = pipeline("depth-estimation", model=args.depth_model, device=dev)

    summary = {
        "total_images": len(image_files),
        "successful": 0,
        "failed": 0,
        "results": [],
    }

    t0_all = time.time()

    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Memproses: {img_path.name}")
        t_start = time.time()

        item_folder = out_dir / sanitize_folder_name(img_path.stem, index=idx)
        item_folder.mkdir(parents=True, exist_ok=True)

        try:
            # A. Buka gambar
            img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
            rgb = np.array(img)
            H, W = rgb.shape[:2]

            # B. Deteksi landmark wajah 478 titik
            lm = get_landmarks(rgb, args.landmarker)

            # C. Inferensi depth map
            out_pipe = pipe(img)
            pred = out_pipe.get("predicted_depth", None)
            if pred is None and "depth" in out_pipe:
                pred = np.array(out_pipe["depth"], dtype=np.float32)

            if hasattr(pred, "squeeze"):
                pred = pred.squeeze()
            if hasattr(pred, "cpu"):
                pred = pred.cpu()
            if hasattr(pred, "numpy"):
                d = pred.numpy().astype(np.float32)
            else:
                d = np.asarray(pred, dtype=np.float32)
            depth = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)

            # D. Skala IPD
            iris_a = lm[rc.IRIS_CENTER_A, :2] * [W, H]
            iris_b = lm[rc.IRIS_CENTER_B, :2] * [W, H]
            mmpp = rc.mm_per_pixel_from_ipd(iris_a, iris_b, args.ipd_mm)
            pts, h_lm = rc.landmarks_to_px_and_height(lm, W, H, mmpp)
            if args.z_scale != 1.0:
                h_lm = h_lm * args.z_scale

            # E. Penyelarasan Affine (A2: Regional Weighted Fit atau Standar)
            nose_pt = pts[1]
            if args.regional_weights:
                prior_w = ri.regional_weights(pts, nose_pt, edges=(0.4, 0.75), weights=(1.0, 0.8, 0.25))
                fit = ri.fit_affine_weighted_robust(rc.sample_map(depth, pts), h_lm, prior_weights=prior_w)
            else:
                fit = rc.fit_affine_robust(rc.sample_map(depth, pts), h_lm)

            if fit["a"] <= 0:
                fit["a"] = abs(fit["a"])
            aligned = fit["a"] * depth + fit["b"]

            # Hitung metrik evaluasi diagnostik
            nrmse_val = ri.normalized_rmse(fit["rmse_mm"], h_lm)
            resid = h_lm - (fit["a"] * rc.sample_map(depth, pts) + fit["b"])
            rings = ri.residuals_by_ring(pts, resid, nose_pt)

            # F. Hitung parameter skala fisik (mm -> px)
            sigma_px = ri.sigma_mm_to_px(args.sigma_mm, mmpp) if args.sigma_mm > 0 else args.sigma
            feather_px = ri.sigma_mm_to_px(args.feather_mm, mmpp) if args.feather_mm > 0 else args.feather

            # Fusi frekuensi
            low = rc.lowfreq_from_landmarks(pts, h_lm, (H, W), sigma_px)
            fused = rc.fuse_frequencies(low, aligned, sigma_px, args.w_detail)

            # G. Bounding box & Crop
            x0, y0 = np.floor(pts.min(0) - args.margin).astype(int).clip(0)
            x1, y1 = np.ceil(pts.max(0) + args.margin).astype(int)
            x1, y1 = min(x1, W), min(y1, H)
            sl = (slice(y0, y1), slice(x0, x1))
            crop_w = x1 - x0
            crop_h = y1 - y0

            s = min(1.0, args.out_width / crop_w)
            target_w = int(crop_w * s)
            target_h = int(crop_h * s)
            fused_c = cv2.resize(fused[sl].astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_AREA)
            pts_c = (pts - [x0, y0]) * s

            # H. Masking & Kompresi
            hull_pts = rc.convex_hull_polygon(pts_c)
            poly_m = rc.polygon_mask(fused_c.shape, hull_pts)
            alpha = rc.soft_mask(poly_m, feather_px)
            relief = rc.compress_relief(fused_c, alpha, args.target_mm, args.gamma)

            # I. Generasi Solid Mesh
            final_pixel_mm = mmpp / s
            tris = rc.heightmap_to_solid(relief, pixel_mm=final_pixel_mm, base_mm=args.base_mm)
            info = rc.check_mesh(tris)

            # J. Simpan berkas-berkas keluaran
            stl_path = item_folder / "relief.stl"
            rc.write_binary_stl(str(stl_path), tris)

            # Visualisasi Landmark
            lm_path = item_folder / "face_landmarks_preview.png"
            save_landmark_visualization(rgb, pts, lm_path)

            # Visualisasi Heightmap
            preview_norm = np.clip(relief / args.target_mm * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(item_folder / "relief_heightmap_preview.png"), preview_norm)

            # Visualisasi Depth Colormap
            d_min, d_max = float(depth.min()), float(depth.max())
            if d_max > d_min:
                d_norm = ((depth - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
                d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
                cv2.imwrite(str(item_folder / "depth_colormap_preview.png"), d_color)

            # Laporan per item lengkap dengan metrik kualitas
            report_item = {
                "file": img_path.name,
                "folder": item_folder.name,
                "stl": str(stl_path),
                "mm_per_px": round(float(mmpp), 4),
                "fit": fit,
                "nrmse": round(float(nrmse_val), 4),
                "core_rmse_mm": round(float(rings["inti"][1]), 3),
                "edge_rmse_mm": round(float(rings["tepi"][1]), 3),
                "mesh": info,
                "duration_sec": round(time.time() - t_start, 2),
            }
            with open(item_folder / "report.json", "w", encoding="utf-8") as f:
                json.dump(report_item, f, indent=2)

            summary["successful"] += 1
            summary["results"].append({"status": "SUCCESS", **report_item})
            print(f"    [✓] Sukses dalam {report_item['duration_sec']} detik -> nRMSE={nrmse_val:.3f}, Inti={rings['inti'][1]:.2f}mm, Tepi={rings['tepi'][1]:.2f}mm")

        except Exception as e:
            err_msg = str(e)
            summary["failed"] += 1
            summary["results"].append({
                "file": img_path.name,
                "status": "FAILED",
                "error": err_msg,
            })
            print(f"    [X] GAGAL: {err_msg}")

    total_time = round(time.time() - t0_all, 2)
    summary["total_time_sec"] = total_time

    # Simpan rekapitulasi batch JSON
    with open(out_dir / "batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Buat tabel ringkasan metrik kualitas (quality_summary.csv)
    rows = []
    for r in summary["results"]:
        if r.get("status") == "SUCCESS":
            fit_data = r.get("fit", {})
            mesh_data = r.get("mesh", {})
            rows.append({
                "folder": r.get("folder", ""),
                "a": round(float(fit_data.get("a", 0.0)), 3),
                "rmse_mm": round(float(fit_data.get("rmse_mm", 0.0)), 3),
                "nrmse": r.get("nrmse", 0.0),
                "core_mm": r.get("core_rmse_mm", 0.0),
                "edge_mm": r.get("edge_rmse_mm", 0.0),
                "inlier": round(float(fit_data.get("inlier_frac", 0.0)), 3),
                "mm_per_px": round(float(r.get("mm_per_px", 0.0)), 3),
                "volume": round(float(mesh_data.get("volume_mm3", 0.0)), 1),
                "watertight": mesh_data.get("watertight", False),
            })

    csv_path = out_dir / "quality_summary.csv"
    if rows:
        try:
            import pandas as pd
            df = pd.DataFrame(rows).sort_values("folder")
            df["flag"] = (df["a"] <= 0) | (df["nrmse"] > 0.25) | (df["inlier"] < 0.6)
            df.to_csv(csv_path, index=False)
            print("\n" + "=" * 95)
            print(" TABEL METRIK KUALITAS RELIEF (Flag aktif jika a<=0 atau nrmse>0.25 atau inlier<0.6):")
            print("=" * 95)
            print(df.to_string(index=False))
            print("=" * 95)
        except Exception:
            # Fallback jika pandas tidak tersedia
            import csv
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["folder", "a", "rmse_mm", "nrmse", "core_mm", "edge_mm", "inlier", "mm_per_px", "volume", "watertight", "flag"])
                writer.writeheader()
                for row in rows:
                    row["flag"] = (row["a"] <= 0) or (row.get("nrmse", 0) > 0.25) or (row["inlier"] < 0.6)
                    writer.writerow(row)

    print("\n" + "=" * 60)
    print(" REKAPITULASI BATCH SELESAI:")
    print(f" Total Foto Diproses : {summary['total_images']}")
    print(f" Berhasil             : {summary['successful']}")
    print(f" Gagal                : {summary['failed']}")
    print(f" Total Waktu          : {total_time} detik")
    print(f" Summary CSV          : {csv_path}")
    print(f" Laporan Lengkap      : {out_dir / 'batch_summary.json'}")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description="Batch Processing Foto Wajah ke Relief STL 3D")
    p.add_argument("--input-dir", default="data/my_photos", help="Folder sumber foto wajah")
    p.add_argument("--out-dir", default="out/batch_results", help="Folder tujuan hasil STL")
    p.add_argument("--landmarker", default="models/face_landmarker.task", help="Path model Face Landmarker")
    p.add_argument(
        "--depth-model",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Model ID HuggingFace Depth Anything V2",
    )
    p.add_argument("--device", default="auto", help="auto, cuda, atau cpu")
    p.add_argument("--ipd-mm", type=float, default=63.0, help="Jarak antar pupil riil (mm)")
    p.add_argument("--z-scale", type=float, default=1.0, help="Faktor pengali kedalaman landmark (menambah mancung hidung/profil)")
    p.add_argument("--sigma-mm", type=float, default=3.0, help="Skala Gaussian fusi dalam mm (standar fisik konsisten antar resolusi)")
    p.add_argument("--feather-mm", type=float, default=5.0, help="Kelembutan batas luar wajah dalam mm")
    p.add_argument("--sigma", type=float, default=6.0, help="Fallback sigma dalam piksel jika sigma-mm <= 0")
    p.add_argument("--w-detail", type=float, default=1.4, help="Bobot detail kedalaman mikro (Depth Anything)")
    p.add_argument("--feather", type=float, default=8.0, help="Fallback feather dalam piksel jika feather-mm <= 0")
    p.add_argument("--regional-weights", action="store_true", default=True, help="Aktifkan pembobotan regional (kurangi bobot tepi rahang)")
    p.add_argument("--no-regional-weights", dest="regional_weights", action="store_false", help="Nonaktifkan pembobotan regional")
    p.add_argument("--margin", type=float, default=25.0, help="Margin luar crop wajah (px)")
    p.add_argument("--out-width", type=int, default=384, help="Resolusi lebar heightmap")
    p.add_argument("--target-mm", type=float, default=10.0, help="Tinggi maksimal tonjolan relief (mm)")
    p.add_argument("--gamma", type=float, default=0.8, help="Pangkat gamma kurva relief")
    p.add_argument("--base-mm", type=float, default=2.0, help="Ketebalan alas dasar plakat (mm)")
    return p.parse_args()


if __name__ == "__main__":
    run_batch(parse_args())
