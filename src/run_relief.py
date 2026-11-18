"""Pipeline Utama: Foto Wajah -> Landmark MediaPipe + Depth Anything V2 -> STL Relief 3D.

Dijalankan secara langsung melalui CLI:
python src/run_relief.py path/ke/foto.jpg --out out/hasil
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

# Tambahkan src ke path jika diimpor langsung
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import relief_core as rc
    import relief_improve as ri
except ImportError:
    from src import relief_core as rc
    from src import relief_improve as ri


def get_landmarks(rgb: np.ndarray, model_path: str) -> np.ndarray:
    """Deteksi 478 titik landmark 3D ternormalisasi [0, 1] menggunakan MediaPipe FaceLandmarker."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"File model MediaPipe '{model_path}' tidak ditemukan.\n"
            f"Jalankan: python models/download_models.py untuk mengunduh model face_landmarker.task."
        )

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
    except ImportError as e:
        raise ImportError(
            "Pustaka 'mediapipe' belum terinstal. Install dengan: pip install mediapipe"
        ) from e

    base_options = mpp.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    with vision.FaceLandmarker.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_image)
        if not res.face_landmarks:
            raise RuntimeError("Wajah tidak terdeteksi dalam foto. Pastikan wajah terlihat jelas dan tegak frontal.")
        # Ambil wajah pertama: bentuk (478, 3)
        return np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], dtype=np.float64)


def get_depth(pil_img: Image.Image, model_id: str, device: str = "auto") -> np.ndarray:
    """Inferensi depth estimation menggunakan Hugging Face transformers pipeline."""
    try:
        import torch
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            "Pustaka 'torch' atau 'transformers' belum terinstal. "
            "Install dengan: pip install torch torchvision transformers"
        ) from e

    if device == "auto":
        dev = 0 if torch.cuda.is_available() else -1
    elif device.lower() in ("cuda", "gpu", "0"):
        dev = 0
    else:
        dev = -1

    device_str = "GPU (CUDA)" if dev == 0 else "CPU"
    print(f"[*] Menjalankan depth estimation dengan model '{model_id}' pada {device_str}...")

    pipe = pipeline("depth-estimation", model=model_id, device=dev)
    out = pipe(pil_img)

    # Ekstraksi predicted_depth secara robust
    pred = out.get("predicted_depth", None)
    if pred is None:
        if "depth" in out and isinstance(out["depth"], Image.Image):
            pred = np.array(out["depth"], dtype=np.float32)
        else:
            raise RuntimeError("Gagal mengekstrak peta depth dari keluaran model.")

    if hasattr(pred, "squeeze"):
        pred = pred.squeeze()
    if hasattr(pred, "cpu"):
        pred = pred.cpu()
    if hasattr(pred, "numpy"):
        d = pred.numpy().astype(np.float32)
    else:
        d = np.asarray(pred, dtype=np.float32)

    # Resize bicubic ke ukuran piksel asli foto
    W, H = pil_img.size
    return cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)


def create_mock_face(W: int = 600, H: int = 800) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Menghasilkan data sintetis untuk pengujian dry-run pipeline tanpa perlu bobot AI."""
    print("[!] Mode sintetis / mock aktif (dry-run tanpa model AI)")
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = W / 2, H / 2
    ell = ((xx - cx) / (W * 0.35)) ** 2 + ((yy - cy) / (H * 0.40)) ** 2
    head = np.clip(1 - ell, 0, None) ** 0.5 * 30.0
    nose = 14.0 * np.exp(-(((xx - cx) / 12) ** 2 + ((yy - (cy - 10)) / 22) ** 2))
    depth_sim = head + nose + 5.0

    # 478 pseudo-landmark sintetis
    rng = np.random.default_rng(42)
    u_x = rng.uniform(cx - W * 0.25, cx + W * 0.25, 478)
    u_y = rng.uniform(cy - H * 0.30, cy + H * 0.30, 478)

    # Titik iris
    u_x[468], u_y[468] = cx - 40, cy - 40
    u_x[473], u_y[473] = cx + 40, cy - 40

    lm = np.zeros((478, 3), dtype=np.float64)
    lm[:, 0] = u_x / W
    lm[:, 1] = u_y / H
    # z negatif = lebih dekat ke kamera
    pts_px = np.stack([u_x, u_y], axis=1)
    h_sample = rc.sample_map(depth_sim, pts_px)
    lm[:, 2] = -h_sample / (W * 0.5)

    rgb = np.full((H, W, 3), 200, dtype=np.uint8)
    return rgb, lm, depth_sim


def save_landmark_visualization(rgb_img: np.ndarray, pts_xy: np.ndarray, out_path: Path):
    """Menggambar 478 titik landmark wajah, pusat iris, dan kontur batas pada citra."""
    vis_bgr = cv2.cvtColor(rgb_img.copy(), cv2.COLOR_RGB2BGR)

    # 1. Gambar batas luar convex hull
    try:
        hull = rc.convex_hull_polygon(pts_xy).astype(np.int32)
        cv2.polylines(vis_bgr, [hull], isClosed=True, color=(255, 200, 0), thickness=2, lineType=cv2.LINE_AA)
    except Exception:
        pass

    # 2. Gambar 478 titik landmark
    for i, (px, py) in enumerate(pts_xy):
        pt = (int(round(px)), int(round(py)))
        if i in (rc.IRIS_CENTER_A, rc.IRIS_CENTER_B):
            # Iris mata: lingkaran merah tebal + pusat kuning
            cv2.circle(vis_bgr, pt, 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(vis_bgr, pt, 2, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        else:
            # Titik normal landmark: hijau
            cv2.circle(vis_bgr, pt, 1, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    # Garis pengukur skala IPD antar iris
    ia = (int(round(pts_xy[rc.IRIS_CENTER_A, 0])), int(round(pts_xy[rc.IRIS_CENTER_A, 1])))
    ib = (int(round(pts_xy[rc.IRIS_CENTER_B, 0])), int(round(pts_xy[rc.IRIS_CENTER_B, 1])))
    cv2.line(vis_bgr, ia, ib, (0, 0, 255), 1, lineType=cv2.LINE_AA)

    cv2.imwrite(str(out_path), vis_bgr)



def run_pipeline(args: argparse.Namespace) -> dict:
    """Eksekusi pipeline lengkap dari foto hingga file STL."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Buka dan normalisasi citra
    if args.synthetic:
        rgb, lm, depth = create_mock_face()
        img = Image.fromarray(rgb)
        H, W = rgb.shape[:2]
    else:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"File foto '{args.image}' tidak ditemukan.")
        img = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
        rgb = np.array(img)
        H, W = rgb.shape[:2]
        print(f"[*] Foto termuat: {W}x{H} piksel")

        # 2. Deteksi Landmark
        print(f"[*] Mendeteksi landmark wajah menggunakan {args.landmarker}...")
        lm = get_landmarks(rgb, args.landmarker)

        # 3. Depth estimation
        print(f"[*] Mengestimasi peta kedalaman...")
        depth = get_depth(img, args.depth_model, device=args.device)

    # 4. Skala mm per piksel dari IPD
    iris_a = lm[rc.IRIS_CENTER_A, :2] * [W, H]
    iris_b = lm[rc.IRIS_CENTER_B, :2] * [W, H]
    mmpp = rc.mm_per_pixel_from_ipd(iris_a, iris_b, args.ipd_mm)
    print(f"[*] Jarak iris dihitung. Skala fisik: {mmpp:.4f} mm/piksel (IPD={args.ipd_mm} mm)")

    # 5. Konversi landmark ke koordinat piksel dan ketinggian mm
    pts, h_lm = rc.landmarks_to_px_and_height(lm, W, H, mmpp)
    if args.z_scale != 1.0:
        h_lm = h_lm * args.z_scale
        print(f"[*] z_scale diterapkan: {args.z_scale:.2f}")

    # 6. Robust Affine Alignment (A2: Regional Weighted Fit atau Standar)
    d_samples = rc.sample_map(depth, pts)
    nose_pt = pts[1]
    if args.regional_weights:
        prior_w = ri.regional_weights(pts, nose_pt, edges=(0.4, 0.75), weights=(1.0, 0.8, 0.25))
        fit = ri.fit_affine_weighted_robust(d_samples, h_lm, prior_weights=prior_w)
    else:
        fit = rc.fit_affine_robust(d_samples, h_lm)

    nrmse_val = ri.normalized_rmse(fit["rmse_mm"], h_lm)
    print(
        f"[*] Penyelarasan Affine: a={fit['a']:.4f}, b={fit['b']:.4f}, "
        f"RMSE={fit['rmse_mm']:.3f} mm (nRMSE={nrmse_val:.3f}), inlier={fit['inlier_frac'] * 100:.1f}%"
    )

    if fit["a"] <= 0:
        msg = f"PERINGATAN: Nilai kemiringan affine a <= 0 ({fit['a']:.4f}). Polaritas depth mungkin terbalik!"
        if args.auto_fix_polarity:
            print(f"[!] {msg} Mengoreksi tanda secara otomatis...")
            fit["a"] = abs(fit["a"])
        else:
            print(f"[WARN] {msg}")

    aligned = fit["a"] * depth + fit["b"]

    # 7. Bentuk kasar & Fusi frekuensi (Skala Fisik dalam mm)
    sigma_px = ri.sigma_mm_to_px(args.sigma_mm, mmpp) if args.sigma_mm > 0 else args.sigma
    feather_px = ri.sigma_mm_to_px(args.feather_mm, mmpp) if args.feather_mm > 0 else args.feather

    print(f"[*] Fusi frekuensi: sigma={args.sigma_mm:.1f} mm ({sigma_px:.1f} px), w_detail={args.w_detail}...")
    low = rc.lowfreq_from_landmarks(pts, h_lm, (H, W), sigma_px)
    fused = rc.fuse_frequencies(low, aligned, sigma_px, args.w_detail)

    # 8. Bounding box wajah dengan margin
    x0, y0 = np.floor(pts.min(axis=0) - args.margin).astype(int).clip(0)
    x1, y1 = np.ceil(pts.max(axis=0) + args.margin).astype(int)
    x1, y1 = min(x1, W), min(y1, H)
    sl = (slice(y0, y1), slice(x0, x1))
    crop_w = x1 - x0
    crop_h = y1 - y0

    # Skala ulang crop ke out_width yang ditentukan
    s = min(1.0, args.out_width / crop_w)
    target_w = int(crop_w * s)
    target_h = int(crop_h * s)
    fused_c = cv2.resize(fused[sl].astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_AREA)
    pts_c = (pts - [x0, y0]) * s

    # 9. Masking wajah halus
    hull_pts = rc.convex_hull_polygon(pts_c)
    poly_m = rc.polygon_mask(fused_c.shape, hull_pts)
    alpha = rc.soft_mask(poly_m, feather_px)

    # 10. Kompresi relief ke target_mm
    print(f"[*] Kompresi relief: target={args.target_mm} mm, gamma={args.gamma}...")
    relief = rc.compress_relief(fused_c, alpha, target_mm=args.target_mm, gamma=args.gamma)

    # Skala piksel akhir setelah resize crop
    final_pixel_mm = mmpp / s

    # 11. Generasi Mesh Padat (Watertight Solid)
    print(f"[*] Membangun solid watertight mesh (base={args.base_mm} mm)...")
    tris = rc.heightmap_to_solid(relief, pixel_mm=final_pixel_mm, base_mm=args.base_mm)
    info = rc.check_mesh(tris)

    print("\n" + "=" * 50)
    print(f" HASIL PEMERIKSAAN MESH:")
    print(f" - Watertight (kedap air) : {'YA (VALID)' if info['watertight'] else 'TIDAK (CACAT)'}")
    print(f" - Jumlah Segitiga        : {info['n_triangles']:,}")
    print(f" - Volume Mesh            : {info['volume_mm3']:.1f} mm³")
    print(
        f" - Dimensi Fisik (BBox)   : X={info['bbox_mm'][0]:.1f} mm, "
        f"Y={info['bbox_mm'][1]:.1f} mm, Z={info['bbox_mm'][2]:.1f} mm"
    )
    print("=" * 50 + "\n")

    # 12. Simpan file STL dan Laporan
    stl_path = out_dir / "relief.stl"
    report_path = out_dir / "report.json"
    rc.write_binary_stl(str(stl_path), tris)
    print(f"[✓] File STL berhasil disimpan: {stl_path}")

    # Simpan peta ketinggian normalisasi sebagai preview visual PNG
    preview_norm = np.clip(relief / args.target_mm * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "relief_heightmap_preview.png"), preview_norm)

    # Simpan visualisasi 478 landmark wajah
    lm_preview_path = out_dir / "face_landmarks_preview.png"
    save_landmark_visualization(rgb, pts, lm_preview_path)
    print(f"[✓] Visualisasi landmark wajah disimpan: {lm_preview_path}")

    # Simpan visualisasi colormap depth map
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max > d_min:
        d_norm = ((depth - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(out_dir / "depth_colormap_preview.png"), d_color)

    report_data = {
        "args": vars(args),
        "mm_per_px_original": float(mmpp),
        "mm_per_px_scaled": float(final_pixel_mm),
        "fit": fit,
        "mesh": info,
        "bounding_box_mm": info["bbox_mm"],
        "outputs": {
            "stl": str(stl_path),
            "heightmap_preview": str(out_dir / "relief_heightmap_preview.png"),
            "landmarks_preview": str(lm_preview_path),
            "depth_colormap_preview": str(out_dir / "depth_colormap_preview.png"),
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[✓] Laporan evaluasi disimpan: {report_path}")

    return report_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Konversi Foto Wajah 2D ke Relief 3D STL Siap Cetak",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image", nargs="?", default="", help="Path ke file foto wajah (JPG/PNG)")
    p.add_argument("--out", default="out/sample", help="Direktori penyimpanan file keluaran")
    p.add_argument("--landmarker", default="models/face_landmarker.task", help="Path model Face Landmarker")
    p.add_argument(
        "--depth-model",
        dest="depth_model",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="HuggingFace model ID untuk Depth Anything V2 (Small/Base/Large)",
    )
    p.add_argument("--device", default="auto", help="Device inferensi: auto, cuda, atau cpu")
    p.add_argument("--ipd-mm", dest="ipd_mm", type=float, default=63.0, help="Jarak antar pupil riil (mm)")
    p.add_argument("--z-scale", dest="z_scale", type=float, default=1.0, help="Faktor kalibrasi kedalaman landmark")
    p.add_argument("--sigma-mm", dest="sigma_mm", type=float, default=3.0, help="Skala Gaussian fusi dalam mm (standar fisik)")
    p.add_argument("--feather-mm", dest="feather_mm", type=float, default=5.0, help="Kelembutan batas luar wajah dalam mm")
    p.add_argument("--sigma", type=float, default=6.0, help="Fallback sigma dalam piksel jika sigma-mm <= 0")
    p.add_argument("--w-detail", dest="w_detail", type=float, default=1.4, help="Bobot kontribusi detail mikro")
    p.add_argument("--feather", type=float, default=8.0, help="Fallback feather dalam piksel jika feather-mm <= 0")
    p.add_argument("--regional-weights", action="store_true", default=True, help="Aktifkan pembobotan regional (inti 1.0, tengah 0.8, tepi 0.25)")
    p.add_argument("--no-regional-weights", dest="regional_weights", action="store_false", help="Nonaktifkan pembobotan regional")
    p.add_argument("--margin", type=float, default=25.0, help="Margin luar crop wajah (px)")
    p.add_argument("--out-width", dest="out_width", type=int, default=384, help="Resolusi lebar grid STL")
    p.add_argument("--target-mm", dest="target_mm", type=float, default=10.0, help="Tinggi maksimal tonjolan relief (mm)")
    p.add_argument("--gamma", type=float, default=0.8, help="Pangkat gamma kurva relief (<1 mengangkat area rendah)")
    p.add_argument("--base-mm", dest="base_mm", type=float, default=2.0, help="Ketebalan alas plakat dasar (mm)")
    p.add_argument(
        "--auto-fix-polarity",
        action="store_true",
        default=True,
        help="Otomatis balikkan tanda jika a <= 0 terdeteksi",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Gunakan wajah sintetis untuk pengujian kering tanpa model AI",
    )
    return p.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if not cli_args.synthetic and not cli_args.image:
        print("Error: Harap sertakan path gambar foto atau gunakan flag --synthetic untuk demo.")
        sys.exit(1)
    run_pipeline(cli_args)
