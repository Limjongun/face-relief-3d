# Blueprint Paket 1 — Landmark Wajah → Relief 3D-Printable

> Proyek eksplorasi murni (tanpa target jurnal). Tujuan: dari **satu foto wajah** menghasilkan **file STL yang bisa dicetak** (relief/bas-relief), plus varian lithophane dan patung wireframe landmark.
> Model: **MediaPipe Face Landmarker** (bentuk kasar + skala) + **Depth Anything V2** (detail halus). Opsional: MoGe-2 / Depth Pro untuk skala metrik.

**Status verifikasi (jujur):**

| Bagian | Status |
|---|---|
| `relief_core.py` (skala IPD, fit affine robust, fusi frekuensi, mask, kompres relief, mesh padat, STL, cek watertight, lithophane) | **Sudah diuji** pada data sintetis: 12 tes pytest lulus |
| Skeleton integrasi MediaPipe + Depth Anything (Bagian 7) | **Belum diuji**. Sandbox tidak punya jaringan, torch, atau transformers. Anggap sebagai titik awal, bukan kode final |
| Angka parameter default (Bagian 8) | Titik awal dari desain, **belum dituning pada foto asli**. Eksperimen E1–E10 gunanya untuk itu |
| URL model dan nama repo HF | Dari ingatan saya, **verifikasi dulu** sebelum download |

---

## 1. Ringkasan

### 1.1 Gambaran

```
foto wajah ──► landmark 478 titik ──► tinggi kasar (mm) ─┐
     │                                                    ├─► fusi ─► mask ─► kompres relief ─► heightmap ─► mesh padat ─► STL
     └──────► Depth Anything V2 ──► depth relatif ─► align affine (robust) ─┘
```

Ide inti: **landmark tahu bentuk global tetapi jarang (478 titik)**, **depth model rapat tetapi hanya relatif (tidak ada satuan, tidak tahu skala)**. Digabung: skala dan bentuk kasar dari landmark, detail halus (lipatan, tekstur kulit, alis) dari depth.

### 1.2 Keluaran

| Keluaran | Isi | Prioritas |
|---|---|---|
| **A. Relief wajah (STL)** | Plakat padat, wajah timbul 8–12 mm di atas alas 2 mm | Utama |
| **B. Lithophane (STL)** | Plat tipis; ketebalan mengikuti kecerahan foto, terlihat saat disinari dari belakang | Varian mudah |
| **C. Patung wireframe landmark** | Bola + batang di 478 titik, dari triangulasi mesh MediaPipe | Varian kreatif |
| D. Laporan metrik | Residu landmark-vs-depth (mm), status watertight, volume, bbox | Wajib untuk evaluasi |

### 1.3 Di luar cakupan (sengaja)

- Bukan rekonstruksi 3D penuh (kepala 360°, belakang kepala). Itu Paket 2/3 (FLAME/MICA/DECA).
- Bukan akurat secara metrik absolut. Skala hanya diperkirakan dari IPD rata-rata 63 mm (error orang per orang bisa ±5–10%).
- Bukan pengganti 3D scan.

---

## 2. Cakupan dan ekspektasi hasil

| Pertanyaan | Jawaban realistis |
|---|---|
| Apakah hasil mirip orangnya? | Ya untuk foto frontal berpencahayaan baik; kemiripan dari kontur mata, hidung, alis, bibir. Rahang dan pipi lebih lemah |
| Apakah hidung punya tinggi yang benar? | Dari landmark ya kasar (z MediaPipe relatif dan diturunkan dari model, bukan ukuran nyata). Tinggi hidung bisa meleset beberapa mm |
| Apakah bisa dicetak tanpa support? | Ya. Relief dicetak **menghadap atas** di atas bidang datar (alas rata). Tidak ada overhang berarti |
| Foto apa yang gagal? | Profil samping > 30°, wajah tertutup masker/rambut, kacamata gelap, flash keras, wajah sangat kecil di frame |

---

## 3. Setup

### 3.1 Perangkat

| Komponen | Minimum | Disarankan |
|---|---|---|
| OS | Windows/Linux/macOS | Ubuntu 22/24 atau WSL |
| Python | 3.10+ | 3.11 |
| GPU | Tidak wajib (CPU lambat tapi jalan untuk model Small/Base) | NVIDIA ≥ 6 GB untuk Depth Anything V2 Large |
| Printer | FDM biasa | Resin untuk detail halus |

### 3.2 Paket

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pillow opencv-python mediapipe pytest
pip install torch transformers           # untuk depth model
# opsional
pip install trimesh                      # validasi/perbaikan mesh pihak ketiga (tidak dipakai core)
```

### 3.3 Model

| Model | Sumber | Catatan |
|---|---|---|
| Face Landmarker | file `face_landmarker.task` dari hub model MediaPipe (`storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`) | **Verifikasi URL**. Simpan di `models/` |
| Depth Anything V2 | HF: `depth-anything/Depth-Anything-V2-Small-hf` / `Base-hf` / `Large-hf` | **Lisensi berbeda**: Small umumnya Apache-2.0, Base/Large/Giant non-komersial (CC-BY-NC-4.0). Cek model card. Karena ini proyek pribadi, tidak masalah; jangan dijual hasilnya sebagai produk tanpa cek lisensi |
| Depth Anything V3 | Rilis lanjutan keluarga yang sama | Opsional. Cek ketersediaan, API, dan lisensi sebelum dipakai |
| MoGe-2 / Depth Pro | repo masing-masing | Opsional (Bagian 9, E8). Keluarannya metrik/affine-invariant dengan skala lebih baik, untuk dibandingkan dengan skala IPD |

---

## 4. Protokol pengambilan foto

Kualitas foto menentukan 70% hasil. Untuk foto sendiri:

| Faktor | Aturan |
|---|---|
| Sudut | Frontal, yaw/pitch < 10°. Kamera sejajar mata |
| Jarak | 1–1,5 m dengan zoom 2× (hindari lensa lebar dekat: distorsi memperbesar hidung) |
| Cahaya | Lembut dan dari samping-depan 45° (jendela). Hindari flash langsung dan backlight |
| Ekspresi | Netral, mulut tertutup |
| Penghalang | Rambut jangan menutup dahi/alis; lepas kacamata |
| Resolusi | Wajah ≥ 600 px lebar |
| Latar | Polos jika bisa (membantu depth model) |
| Kalibrasi skala (opsional) | Ukur jarak antar pupil sebenarnya (mis. dengan penggaris di cermin atau data dari optik) lalu isi `ipd_mm` |

Dataset tambahan untuk variasi (tanpa niat jurnal; cek lisensi masing-masing): FFHQ, CelebA-HQ. Pakai untuk uji ketahanan pipeline, bukan untuk melatih apa pun.

**Privasi:** foto orang lain hanya dengan izin. Jangan unggah foto orang lain ke layanan pihak ketiga. Semua pipeline ini berjalan lokal.

---

## 5. Tahapan detail (S1–S11)

### S1. Muat gambar dan normalisasi
- Baca dengan Pillow, terapkan EXIF-orientation (`ImageOps.exif_transpose`), konversi RGB.
- Jika sisi panjang > 2048 px, perkecil (hemat memori), simpan faktor skala.

### S2. Deteksi landmark
- MediaPipe Face Landmarker, `num_faces=1`. Hasil: 478 titik `(x, y, z)`; x,y ternormalisasi [0,1], z relatif dengan skala mirip x (z negatif = lebih dekat kamera).
- Tolak jika 0 wajah atau > 1 wajah (atau pilih terbesar).
- Cek kualitas: wajah tidak terpotong (semua x,y dalam [0,1]).

### S3. Skala mm/piksel dari IPD
- Pusat iris: indeks 468 dan 473. `mm_per_px = ipd_mm / jarak_piksel(468, 473)`.
- `ipd_mm` default 63. Jika pengguna tahu nilai sebenarnya, pakai itu.
- Fungsi: `mm_per_pixel_from_ipd`.

### S4. Tinggi landmark (mm)
- `titik_px = (x·W, y·H)`; `tinggi_mm = -z · W · mm_per_px` (z dikali lebar gambar karena z berskala x).
- Fungsi: `landmarks_to_px_and_height`.
- Catatan: z MediaPipe berasal dari model, bukan pengukuran. Bentuk kasar baik, ukuran mutlak tidak dijamin.

### S5. Depth relatif
- Jalankan Depth Anything V2 pada **gambar penuh**, resize `predicted_depth` ke ukuran gambar (bicubic).
- Keluaran = **disparity relatif** (besar = dekat). Tidak bersatuan.

### S6. Penyelarasan affine robust
- Sampel depth di 478 titik landmark: `d_i`. Cocokkan `h_i ≈ a·d_i + b` memakai IRLS (bobot Tukey bisquare, skala MAD).
- Robust karena beberapa landmark (mulut, kontur rahang, area dahi) sering menyimpang dari depth.
- Kembalikan `a, b, rmse_mm, inlier_frac`. **Peringatan jika `a ≤ 0`** (tanda polaritas depth terbalik atau gagal).
- `depth_aligned = a·depth + b` (sekarang dalam mm).
- Fungsi: `sample_map`, `fit_affine_robust`.

### S7. Bentuk kasar dari landmark
- Interpolasi Delaunay dari 478 titik ke seluruh gambar, di luar hull diisi nearest-neighbour, lalu Gaussian σ ≈ 6 px.
- Fungsi: `lowfreq_from_landmarks`.

### S8. Fusi frekuensi
- `fusi = low_landmark + w · (depth_aligned − Gaussian(depth_aligned, σ))`.
- Intuisi: bentuk besar dipercaya dari landmark; detail halus dari depth; σ menentukan batas antara "besar" dan "halus".
- `w=0` hanya landmark; `w=1` detail penuh; `w>1` detail ditekankan.
- Fungsi: `fuse_frequencies`.

### S9. Mask wajah
- Hull cembung dari semua landmark (atau kontur oval wajah MediaPipe untuk bentuk lebih rapi), lalu feather (smoothstep) 6–12 px agar tepi relief meluncur ke 0, bukan tebing tajam.
- Fungsi: `convex_hull_polygon`, `polygon_mask`, `soft_mask`.

### S10. Kompres relief
- Normalisasi persentil 1–99 di dalam mask, pangkat γ, skala ke `target_mm` (tinggi relief, default 10 mm). Luar mask = 0.
- γ < 1 mengangkat area rendah (detail dangkal lebih terlihat); γ > 1 menekan.
- Fungsi: `compress_relief`.
- Catatan: ini yang membuat relief tipis tetapi masih terbaca. Relief 10 mm dari wajah ~15 cm adalah kompresi besar (hidung asli menonjol sekitar 25–30 mm).

### S11. Mesh padat dan STL
- Heightmap → solid: permukaan atas `z = base + h`, bidang bawah `z = 0`, 4 dinding sisi. Mesh **watertight**, orientasi normal konsisten (volume positif).
- `pixel_mm = mm_per_px × (lebar_asli / lebar_heightmap)`. **Jika Anda memperkecil crop, skala harus ikut disesuaikan.**
- Jumlah segitiga ≈ `4·(H−1)·(W−1)`. Untuk 384×384 ≈ 590 ribu segitiga (~29 MB STL); 512×512 ≈ 1,04 juta (~52 MB). Slicer modern sanggup, tetapi 256–384 sudah cukup untuk FDM.
- Fungsi: `heightmap_to_solid`, `check_mesh`, `write_binary_stl`.

---

## 6. API modul inti (`src/relief_core.py`)

| Fungsi | Masukan | Keluaran |
|---|---|---|
| `mm_per_pixel_from_ipd(a_xy, b_xy, ipd_mm=63)` | dua pusat iris (px) | mm per piksel |
| `landmarks_to_px_and_height(lm, W, H, mm_per_px)` | landmark (N,3) | `(pts_px, tinggi_mm)` |
| `sample_map(arr, pts)` | peta (H,W), titik | nilai bilinear |
| `fit_affine_robust(d, h)` | depth di titik, tinggi | `dict(a, b, rmse_mm, inlier_frac)` |
| `polygon_mask(shape, poly)` / `convex_hull_polygon(pts)` | | mask bool / poligon |
| `soft_mask(mask, feather_px)` | | alpha 0..1 |
| `lowfreq_from_landmarks(pts, h, shape, sigma)` | | peta kasar (mm) |
| `fuse_frequencies(low, depth_aligned, sigma, w)` | | peta fusi (mm) |
| `compress_relief(h, alpha, target_mm, gamma, p_low, p_high)` | | peta tinggi 0..target (mm) |
| `heightmap_to_solid(h, pixel_mm, base_mm)` | | segitiga `(T,3,3)` |
| `check_mesh(tris)` | | `dict(watertight, volume_mm3, bbox_mm, ...)` |
| `write_binary_stl(path, tris)` / `read_binary_stl(path)` | | file STL biner |
| `lithophane_thickness(gray01, t_min, t_max, gamma)` | | ketebalan (mm) |

Yang dicakup tes (12): skala IPD, tanda tinggi landmark, pemulihan affine dengan 15% outlier (dan bukti OLS biasa lebih buruk), penyelarasan depth, mask dan feather, hull, fusi frekuensi (kontribusi tekstur sama persis dengan high-pass-nya), kompres relief, mesh watertight + volume analitik + jumlah segitiga + ukuran file STL + round-trip, deteksi lubang, end-to-end sintetis, lithophane.

---

## 7. Skeleton integrasi (BELUM DIUJI)

> Ditulis dari pengetahuan API, tanpa bisa menjalankannya di sandbox. Kemungkinan perlu penyesuaian kecil. Jalankan pada satu foto dulu dan periksa tiap peta antara.

```python
# src/run_relief.py
import argparse, json, os
import numpy as np, cv2
from PIL import Image, ImageOps
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
import torch
from transformers import pipeline
import relief_core as rc

def get_landmarks(rgb, model_path):
    opts = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE, num_faces=1)
    det = vision.FaceLandmarker.create_from_options(opts)
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        raise RuntimeError("Wajah tidak terdeteksi")
    return np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]])   # (478,3)

def get_depth(pil_img, model_id):
    pipe = pipeline("depth-estimation", model=model_id,
                    device=0 if torch.cuda.is_available() else -1)
    out = pipe(pil_img)
    d = out["predicted_depth"].squeeze().float().cpu().numpy()          # ukuran internal model
    W, H = pil_img.size
    return cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)         # disparity relatif

def main(a):
    img = ImageOps.exif_transpose(Image.open(a.image)).convert("RGB")
    rgb = np.array(img); H, W = rgb.shape[:2]
    lm = get_landmarks(rgb, a.landmarker)
    iris_a = lm[468, :2] * [W, H]; iris_b = lm[473, :2] * [W, H]
    mmpp = rc.mm_per_pixel_from_ipd(iris_a, iris_b, a.ipd_mm)
    pts, h_lm = rc.landmarks_to_px_and_height(lm, W, H, mmpp)

    depth = get_depth(img, a.depth_model)
    fit = rc.fit_affine_robust(rc.sample_map(depth, pts), h_lm)
    if fit["a"] <= 0: print("PERINGATAN: a<=0, polaritas depth mungkin terbalik")
    aligned = fit["a"] * depth + fit["b"]

    low = rc.lowfreq_from_landmarks(pts, h_lm, (H, W), a.sigma)
    fused = rc.fuse_frequencies(low, aligned, a.sigma, a.w_detail)

    # crop ke bbox wajah + margin, lalu perkecil (skala mm/px ikut berubah!)
    x0, y0 = np.floor(pts.min(0) - a.margin).astype(int).clip(0)
    x1, y1 = np.ceil(pts.max(0) + a.margin).astype(int)
    x1, y1 = min(x1, W), min(y1, H)
    sl = (slice(y0, y1), slice(x0, x1))
    crop_w = x1 - x0
    s = min(1.0, a.out_width / crop_w)
    def rs(m): return cv2.resize(m, (int(crop_w * s), int((y1 - y0) * s)), interpolation=cv2.INTER_AREA)

    fused_c = rs(fused[sl].astype(np.float32))
    pts_c = (pts - [x0, y0]) * s
    alpha = rc.soft_mask(rc.polygon_mask(fused_c.shape, rc.convex_hull_polygon(pts_c)), a.feather)
    relief = rc.compress_relief(fused_c, alpha, a.target_mm, a.gamma)
    tris = rc.heightmap_to_solid(relief, pixel_mm=mmpp / s, base_mm=a.base_mm)

    info = rc.check_mesh(tris)
    os.makedirs(a.out, exist_ok=True)
    rc.write_binary_stl(os.path.join(a.out, "relief.stl"), tris)
    json.dump(dict(fit=fit, mm_per_px=mmpp, mesh=info, args=vars(a)),
              open(os.path.join(a.out, "report.json"), "w"), indent=2)
    print(info, fit)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image"); p.add_argument("--out", default="out")
    p.add_argument("--landmarker", default="models/face_landmarker.task")
    p.add_argument("--depth-model", dest="depth_model", default="depth-anything/Depth-Anything-V2-Large-hf")
    p.add_argument("--ipd-mm", dest="ipd_mm", type=float, default=63.0)
    p.add_argument("--sigma", type=float, default=6.0)
    p.add_argument("--w-detail", dest="w_detail", type=float, default=1.0)
    p.add_argument("--feather", type=float, default=8.0)
    p.add_argument("--margin", type=float, default=20.0)
    p.add_argument("--out-width", dest="out_width", type=int, default=384)
    p.add_argument("--target-mm", dest="target_mm", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--base-mm", dest="base_mm", type=float, default=2.0)
    main(p.parse_args())
```

Hal yang paling mungkin perlu dikoreksi saat pertama dijalankan:

1. Bentuk/dtype `predicted_depth` dari pipeline (squeeze, tipe float).
2. Tanda z: jika relief tampak **terbalik** (hidung menjorok ke dalam), tanda `h_lm` atau polaritas depth perlu dibalik. Peringatan `a ≤ 0` akan membantu mendeteksinya.
3. Satuan z MediaPipe: asumsi z berskala x (dokumentasi menyebut skala hampir sama dengan x). Jika tinggi relief terlalu datar/terlalu tebal secara konsisten, kalibrasi dengan faktor `z_scale` (E2).

### Cara menguji sebelum mencetak
```bash
pytest tests -q                         # modul inti
python src/run_relief.py foto.jpg --out out/
python - <<'EOF'
import relief_core as rc, numpy as np
t = rc.read_binary_stl("out/relief.stl"); print(rc.check_mesh(t))
EOF
```
Buka `relief.stl` di slicer (PrusaSlicer/Cura/Bambu Studio) atau MeshLab dan lihat dengan pencahayaan miring sebelum memutuskan cetak.

---

## 8. Parameter default dan rentang tuning

| Parameter | Default | Rentang | Efek |
|---|---|---|---|
| `ipd_mm` | 63 | 54–74 | Skala semua dimensi X/Y/Z-landmark |
| `sigma` (px, di resolusi foto) | 6 | 3–20 | Batas "bentuk besar vs detail". Besar = lebih banyak detail dari depth, bentuk bisa tidak stabil |
| `w_detail` | 1.0 | 0–2 | Kekuatan detail halus |
| `feather` (px) | 8 | 0–20 | Kelembutan tepi |
| `target_mm` | 10 | 5–20 | Tinggi relief. FDM: ≥ 6 mm agar terbaca |
| `gamma` | 0.8 | 0.5–1.5 | <1 menonjolkan area dangkal |
| `p_low, p_high` | 1, 99 | 0–5 / 95–100 | Pemotong outlier |
| `base_mm` | 2.0 | 1.2–4 | Ketebalan alas. ≥ 1.2 mm agar kokoh |
| `out_width` | 384 | 192–512 | Resolusi heightmap vs ukuran file |
| Ukuran cetak | ~ 100–150 mm lebar | | Atur skala di slicer; relief ikut membesar proporsional, `target_mm` terskala |

> Skala di slicer: bila Anda memperbesar 1,5×, tinggi relief juga 1,5×. Jika ingin lebar 120 mm dengan relief tetap 8 mm, atur `target_mm` setelah menghitung faktor skala, atau skala hanya X/Y di slicer.

---

## 9. Eksperimen (E1–E10)

Format: hipotesis → variabel → metrik → hasil yang dicari. Semua bisa dikerjakan tanpa niat publikasi; catat hasilnya di `experiments/` supaya bisa dibandingkan.

| # | Eksperimen | Hipotesis | Variabel | Metrik |
|---|---|---|---|---|
| **E1** | Ablasi sumber bentuk | Fusi > landmark saja > depth saja | `w_detail`, mode: landmark-only / depth-only / fusi | Residu mm di landmark, penilaian visual (render miring) |
| **E2** | Kalibrasi z MediaPipe | Faktor skala z ≠ 1 meningkatkan kecocokan dengan depth | `z_scale` ∈ {0.5…2} | `rmse_mm` dan `inlier_frac` dari fit affine |
| **E3** | Pilihan σ | Ada σ optimal; terlalu kecil → bentuk "bergelombang", terlalu besar → kehilangan bentuk global | σ ∈ {3,6,10,15,20} | Visual + std dari high-pass |
| **E4** | Ukuran model depth | Large > Base > Small, dengan beda yang mengecil | Small/Base/Large | Residu mm, waktu inferensi |
| **E5** | Skala IPD vs data nyata | IPD 63 menyimpang beberapa % | IPD nyata (ukur) vs 63 | Lebar wajah di STL vs jangka sorong pada kepala/foto referensi |
| **E6** | Pencahayaan | Cahaya samping lembut menghasilkan depth lebih bagus daripada flash | Foto dengan 3–4 kondisi cahaya | Konsistensi relief antar kondisi (RMSE antar relief setelah normalisasi) |
| **E7** | Sudut kepala | Kualitas turun cepat melewati yaw ±20° | Yaw 0/10/20/30° | `rmse_mm`, kegagalan deteksi |
| **E8** | Skala metrik pembanding | Model metrik (MoGe-2/Depth Pro) memberi skala lebih akurat dari IPD | IPD vs model metrik | Selisih lebar/tinggi hidung dalam mm |
| **E9** | Variasi wajah | Pipeline stabil lintas variasi | 30–50 gambar FFHQ/CelebA-HQ (cek lisensi) | Laju kegagalan, distribusi `rmse_mm`, `a>0` rate |
| **E10** | Kompresi relief | γ dan `target_mm` yang berbeda mengubah keterbacaan wajah di cetakan | γ ∈ {0.6,0.8,1.0,1.3}, target ∈ {6,10,15} | Penilaian visual cetak/render; opsional survei ke 3–5 teman "siapa ini?" |

### Metrik evaluasi yang dipakai

| Metrik | Cara hitung | Arti |
|---|---|---|
| Residu landmark–depth | `fit.rmse_mm` | Seberapa konsisten dua sumber bentuk (kecil = sepakat) |
| Inlier fraction | `fit.inlier_frac` | Rendah (< 0,6) → depth atau landmark bermasalah |
| Watertight | `check_mesh` | Wajib `True` sebelum cetak |
| Volume | `volume_mm3` | Estimasi filamen (× densitas) |
| Bounding box | `bbox_mm` | Cek ukuran fisik |
| Fisik | Jangka sorong: lebar antar mata, lebar hidung, tinggi relief vs STL | Ground truth sederhana |
| Stabilitas | RMSE relief antar dua foto orang sama | Variasi akibat foto |

---

## 10. Varian

### 10.1 Lithophane
- Ketebalan tiap piksel: `t = t_min + (t_max − t_min) · (1 − kecerahan)^γ`, dengan `t_min ≈ 0,8 mm`, `t_max ≈ 3 mm`. Terang → tipis (cahaya tembus), gelap → tebal.
- Pakai **gray 0..1** dari foto (setelah CLAHE ringan untuk kontras), lalu `heightmap_to_solid(t - t_min, pixel_mm, base_mm=t_min)` (alas = `t_min`, tinggi tambahan = `t - t_min`, jadi ketebalan total per piksel = `t`).
- Cetak **tegak**, filamen putih, 100% infill atau wall-only; sinari dari belakang (LED hangat).
- Fungsi: `lithophane_thickness`. Mesh lithophane lolos cek watertight pada data acak, tetapi **belum diuji dengan cetakan**.

### 10.2 Patung wireframe landmark (kreatif)
- 478 titik → bola radius 0,8–1,2 mm; sisi triangulasi (tesselation MediaPipe) → batang silinder radius 0,5–0,8 mm.
- Batang tipis berpotensi gagal pada FDM: untuk FDM, radius ≥ 0,8 mm dan gunakan **support** atau cetak menghadap rata (datarkan di bidang). Resin lebih cocok.
- Hubungkan ke alas tipis atau cetak sebagai gantungan.
- Belum diimplementasikan di modul inti. Tahap pengembangan: buat fungsi `tube_mesh` + gabung ke satu mesh (butuh boolean union atau cukup mesh tumpang tindih; slicer menangani tumpang tindih).

---

## 11. Pengaturan cetak

### FDM (relief)

| Parameter | Nilai awal |
|---|---|
| Nozzle / layer | 0,4 mm / 0,12–0,16 mm (lebih halus = detail relief lebih baik) |
| Orientasi | Alas di bed, relief menghadap atas |
| Infill | 15–20% (alas 2 mm sebagian besar solid) |
| Dinding | 3 perimeter |
| Support | Tidak perlu |
| Filamen | PLA matte/abu atau putih; warna terang menonjolkan bayangan |
| Cetak lambat | 40–60 mm/s untuk permukaan atas |
| Ironing | Opsional pada lapisan atas |
| Pasca | Amplas halus, primer, cat tipis; cat gelap lalu dry-brush terang untuk menonjolkan detail |

### Resin (SLA/MSLA)
- Layer 0,03–0,05 mm. Relief orientasi miring ±30–45° agar tidak menempel lebar ke FEP. Cuci dan kuring sesuai resin.

---

## 12. Struktur proyek

```
face-relief/
├─ blueprint_face_relief_3dprint.md
├─ models/                  # face_landmarker.task
├─ data/
│  ├─ my_photos/            # foto sendiri
│  └─ ffhq_sample/          # opsional
├─ src/
│  ├─ relief_core.py        # SUDAH DIUJI
│  └─ run_relief.py         # skeleton Bagian 7 (BELUM DIUJI)
├─ tests/
│  └─ test_relief_core.py   # 12 tes
├─ out/
│  └─ <nama_foto>/
│     ├─ relief.stl
│     └─ report.json
└─ experiments/
   └─ E1_ablasi.md ...
```

---

## 13. Risiko dan pemecahan masalah

| Gejala | Kemungkinan penyebab | Tindakan |
|---|---|---|
| Relief terbalik (hidung masuk) | Polaritas depth atau tanda z | Periksa `fit.a`; jika negatif balik tanda; bandingkan E2 |
| Hasil terlalu datar | `target_mm` kecil / z MediaPipe terlalu kecil / percentil terlalu longgar | Naikkan `target_mm`, turunkan γ, cek E2 |
| Hasil "bergelombang"/berisik | σ terlalu kecil atau `w_detail` terlalu besar | Naikkan σ, turunkan `w_detail`, pakai depth model lebih besar |
| Tepi relief tajam/berundak | Feather terlalu kecil | Naikkan `feather` |
| STL tidak watertight | Seharusnya tidak pada `heightmap_to_solid` (diuji) | Jika terjadi setelah slicer/perbaikan pihak ketiga, bandingkan dengan `check_mesh` |
| Skala fisik meleset | IPD rata-rata ≠ IPD Anda | Ukur IPD, isi `ipd_mm`; atau gunakan E8 |
| Landmark tidak terdeteksi | Wajah kecil, profil, terhalang | Crop foto ke wajah, ulang foto |
| Memori habis | Resolusi terlalu tinggi | Turunkan `out_width`, pakai model depth Small |
| Cetakan mulai terkelupas | Alas tipis atau bed adhesi buruk | `base_mm` ≥ 2, brim |
| Lisensi | Depth Anything V2 Large/Base non-komersial | Proyek pribadi aman; untuk dijual, cek model card atau pakai Small |

### Batasan fundamental
- Depth monokular **bukan ukuran nyata**: detail terlihat meyakinkan tetapi bisa "dikarang" model (mis. hidung terlihat mancung karena prior data).
- Wajah dari 1 foto tidak tahu bentuk belakang/sisi; relief hanya permukaan depan.

---

## 14. Etika dan privasi

- Izin eksplisit untuk foto orang lain. Relief wajah adalah data biometrik turunan.
- Jangan unggah foto/hasil mentah ke layanan pihak ketiga tanpa izin; semua langkah di atas lokal.
- FFHQ/CelebA-HQ memuat wajah orang nyata dengan ketentuan lisensi masing-masing; pakai hanya sesuai ketentuan, jangan dipublikasikan ulang.
- Jangan dipakai untuk meniru identitas orang lain.

---

## 15. Rencana kerja (± 5 hari santai)

| Hari | Target | Keluaran |
|---|---|---|
| 1 | Setup + jalankan tes inti + ambil 5–10 foto sendiri | `pytest` hijau, folder foto |
| 2 | Jalankan `run_relief.py` pada 1 foto, debug integrasi (Bagian 7), periksa tanda/polaritas | `relief.stl` pertama, `report.json` |
| 3 | E1–E3 (ablasi, z-scale, σ), pilih parameter | Tabel parameter terpilih |
| 4 | Cetak percobaan kecil (lebar ~60 mm) + E10 | Cetakan, catatan keterbacaan |
| 5 | Lithophane (10.1), E5/E8 kalibrasi skala, rapikan README | Varian lithophane, catatan akurasi |

Lanjutan opsional: wireframe (10.2), E9 skala besar, lalu pindah ke Paket 2 (FLAME/MICA/DECA) jika ingin kepala 3D utuh.

---

## 16. Checklist

- [ ] `pytest tests -q` hijau
- [ ] `face_landmarker.task` terunduh; URL diverifikasi
- [ ] Model depth terunduh; lisensi dibaca
- [ ] Foto memenuhi protokol (frontal, cahaya lembut)
- [ ] `run_relief.py` berjalan; relief tidak terbalik
- [ ] `fit.a > 0`, `inlier_frac > 0,6`
- [ ] `check_mesh.watertight == True`
- [ ] Preview di slicer/MeshLab (cahaya miring) layak
- [ ] Skala fisik dicek (IPD / jangka sorong)
- [ ] Cetak uji kecil sebelum cetak besar
- [ ] Hasil eksperimen dicatat di `experiments/`

---

## Lampiran A — Rumus

**Skala:** `mm_per_px = IPD_mm / ||iris_468 − iris_473||_px`

**Tinggi landmark:** `h_i = −z_i · W · mm_per_px`

**Fit affine robust (IRLS):** minimalkan `Σ w_i (h_i − a d_i − b)²`, `w_i = (1 − (r_i / (c·s))²)²` bila `|r_i| < c·s`, selain itu 0; `c = 4,685`, `s = 1,4826 · MAD(r)`.

**Fusi:** `F = G_σ(L) + w · (D' − G_σ(D'))`, `D' = a·D + b`, `L` = interpolasi landmark. (Pada kode, `L` sudah dihaluskan σ sebelumnya.)

**Kompres:** `n = clip((F − p_low) / (p_high − p_low), 0, 1)^γ`, `H = target_mm · n · α`

**Mesh padat:** untuk heightmap `H×W`: `4(H−1)(W−1)` segitiga permukaan atas dan bawah ditambah `4((H−1)+(W−1))` segitiga dinding. Volume dari jumlah tetrahedron bertanda: `V = Σ v0 · (v1 × v2) / 6`.

**Lithophane:** `t = t_min + (t_max − t_min) · (1 − g)^γ`, `g ∈ [0,1]` kecerahan.

**Format STL biner:** 80 byte header, `uint32` jumlah segitiga, lalu per segitiga 50 byte (normal 3×float32, 3 titik 9×float32, 2 byte atribut). Ukuran file = `84 + 50 · n`.

---

## Lampiran B — Sumber yang perlu Anda verifikasi sendiri

- Dokumentasi MediaPipe Face Landmarker (skema landmark, indeks iris 468–477, skala z, URL model).
- Model card Depth Anything V2 di Hugging Face (lisensi per ukuran, format keluaran).
- Repo MoGe-2 / Depth Pro bila memakai skala metrik.
- Lisensi FFHQ / CelebA-HQ.
