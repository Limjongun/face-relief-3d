"""Alat diagnostik dan peningkatan akurasi untuk pipeline relief wajah.

Melengkapi relief_core.py (tidak mengubahnya). Hanya numpy + scipy.

Isi
---
* normalized_rmse        : galat fit relatif terhadap rentang tinggi landmark (tidak bergantung skala mm)
* fit_poly_robust        : fit polinomial robust (deg 1/2)
* fit_disparity_model    : h = a/(d+s) + b  (depth model mengeluarkan disparitas, bukan tinggi)
* residuals_by_ring      : residu per cincin jarak dari ujung hidung (inti / tengah / tepi)
* mirror_symmetry_error  : galat simetri kiri-kanan relief (mm)
* shading_consistency    : korelasi antara render Lambert relief dan kecerahan foto
* guided_filter          : haluskan depth dengan menjaga tepi dari gambar panduan
* sigma_mm_to_px         : ubah skala fusi dari mm ke piksel (konsisten antar resolusi)
* resample_heightmap     : samakan jarak piksel (mm/px) antar gambar
* camera_distance_mm     : perkiraan jarak kamera-wajah dari IPD dan fokus (deteksi distorsi perspektif)
* combine_aligned        : agregasi ansambel/TTA beberapa peta kedalaman
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

try:
    import relief_core as rc
except ImportError:
    from src import relief_core as rc


# --------------------------------------------------------------------------- metrik
def normalized_rmse(rmse_mm: float, h_mm, p_low: float = 5, p_high: float = 95) -> float:
    """rmse / rentang tinggi landmark (persentil 5-95). Tidak berubah bila z diskalakan; dipakai untuk membandingkan gambar."""
    h = np.asarray(h_mm, float)
    rng = np.percentile(h, p_high) - np.percentile(h, p_low)
    return float(rmse_mm / rng) if rng > 1e-9 else float("nan")


def _irls(A, h, n_iter=30, c=4.685):
    w = np.ones(len(h))
    coef = np.linalg.lstsq(A, h, rcond=None)[0]
    for _ in range(n_iter):
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(A * sw[:, None], h * sw, rcond=None)[0]
        r = h - A @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = r / (c * s)
        w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        if w.sum() < A.shape[1] + 1:
            break
    r = h - A @ coef
    inl = w > 0
    return coef, r, inl


def regional_weights(pts_xy, center_xy, edges=(0.4, 0.75), weights=(1.0, 0.8, 0.25)) -> np.ndarray:
    """Bobot prior berdasarkan jarak radial ke pusat (ujung hidung): inti (1.0), tengah (0.8), tepi (0.25)."""
    pts = np.asarray(pts_xy, float)
    d = np.linalg.norm(pts - np.asarray(center_xy, float), axis=1)
    d = d / (d.max() + 1e-9)
    w = np.full(len(pts), weights[2], dtype=float)
    w[d < edges[1]] = weights[1]
    w[d < edges[0]] = weights[0]
    return w


def fit_affine_weighted_robust(d, h, prior_weights=None, n_iter: int = 30, c: float = 4.685) -> dict:
    """Fit affine h ~ a*d + b dengan bobot prior regional per wilayah + IRLS Tukey.
    Mengurangi pengaruh distorsi di rahang/tepi wajah tanpa mengorbankan ketahanan outlier."""
    d = np.asarray(d, float)
    h = np.asarray(h, float)
    A = np.stack([d, np.ones_like(d)], axis=1)
    pw = np.ones_like(d) if prior_weights is None else np.asarray(prior_weights, float)
    w = pw.copy()
    coef = np.linalg.lstsq(A * np.sqrt(w)[:, None], h * np.sqrt(w), rcond=None)[0]
    for _ in range(n_iter):
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(A * sw[:, None], h * sw, rcond=None)[0]
        r = h - A @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = r / (c * s)
        tukey = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        w = pw * tukey
        if (w > 0).sum() < 3:
            break
    r = h - A @ coef
    inl = w > 0
    return dict(
        a=float(coef[0]),
        b=float(coef[1]),
        rmse_mm=float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else float("nan"),
        inlier_frac=float(inl.mean()),
    )


def fit_poly_robust(d, h, deg: int = 2) -> dict:
    """h ~ polinomial(d) dengan IRLS Tukey. deg=1 setara fit_affine_robust. Return dict(coef, rmse_mm, inlier_frac)."""
    d = np.asarray(d, float)
    h = np.asarray(h, float)
    A = np.vander(d, deg + 1)
    coef, r, inl = _irls(A, h)
    return dict(
        coef=coef,
        rmse_mm=float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else float("nan"),
        inlier_frac=float(inl.mean()),
    )


def apply_poly(coef, depth):
    return np.polyval(coef, depth)


def fit_disparity_model(d, h, n_grid: int = 40) -> dict:
    """Model h = a / (d + s) + b. d = disparitas relatif dari depth model. Pilih pergeseran s (kandidat dari
    rentang d) dengan rmse robust terkecil. Syarat: d + s > 0 di seluruh data. Return dict(a, b, s, rmse_mm, inlier_frac)."""
    d = np.asarray(d, float)
    h = np.asarray(h, float)
    span = max(d.max() - d.min(), 1e-9)
    best = None
    for s in d.min() * -1 + span * np.geomspace(0.02, 5.0, n_grid):  # d+s > 0 dan melewati rentang cekung-nyaris-linear
        x = 1.0 / (d + s)
        coef, r, inl = _irls(np.stack([x, np.ones_like(x)], 1), h)
        rmse = float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else np.inf
        if best is None or rmse < best["rmse_mm"]:
            best = dict(a=float(coef[0]), b=float(coef[1]), s=float(s), rmse_mm=rmse, inlier_frac=float(inl.mean()))
    return best


def apply_disparity_model(fit: dict, depth):
    return fit["a"] / (np.asarray(depth, float) + fit["s"]) + fit["b"]


def residuals_by_ring(pts_xy, resid, center_xy, edges=(0.4, 0.75)) -> dict:
    """Kelompokkan residu menurut jarak ke pusat (mis. ujung hidung), dinormalkan jarak maksimum.
    Return {"inti": (n, rmse), "tengah": ..., "tepi": ...}. Dipakai untuk melihat di mana galat berkumpul."""
    pts = np.asarray(pts_xy, float)
    r = np.asarray(resid, float)
    d = np.linalg.norm(pts - np.asarray(center_xy, float), axis=1)
    d = d / (d.max() + 1e-9)
    names = ["inti", "tengah", "tepi"]
    bins = [d < edges[0], (d >= edges[0]) & (d < edges[1]), d >= edges[1]]
    return {
        n: (int(b.sum()), float(np.sqrt(np.mean(r[b] ** 2))) if b.any() else float("nan"))
        for n, b in zip(names, bins)
    }


def mirror_symmetry_error(h_mm, mask, axis_x: float) -> float:
    """Rata-rata |h(x) - h(2*axis_x - x)| (mm) di area di mana kedua sisi berada di dalam mask. Wajah tidak simetris sempurna,
    jadi pakai nilai ini untuk MEMBANDINGKAN varian pipeline pada foto yang sama, bukan sebagai nilai absolut."""
    H, W = h_mm.shape
    xs = np.arange(W)
    src = 2 * axis_x - xs
    ok = (src >= 0) & (src <= W - 1)
    mir = np.zeros_like(h_mm)
    mm = np.zeros_like(mask, dtype=bool)
    for y in range(H):
        mir[y] = np.interp(src, xs, h_mm[y], left=0, right=0)
        mm[y] = np.interp(src, xs, mask[y].astype(float), left=0, right=0) > 0.99
    both = mask.astype(bool) & mm & ok[None, :]
    if not both.any():
        return float("nan")
    return float(np.abs(h_mm - mir)[both].mean())


def render_lambert(h_mm, pixel_mm: float, light=(-0.35, -0.35, 0.87)) -> np.ndarray:
    """Render Lambert dari heightmap. Sumbu gambar: x kanan, y BAWAH (baris), z keluar. light = arah ke sumber cahaya."""
    gy, gx = np.gradient(np.asarray(h_mm, float), pixel_mm)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    l = np.asarray(light, float)
    l = l / np.linalg.norm(l)
    return np.clip(n @ l, 0, 1)


def shading_consistency(
    h_mm, pixel_mm: float, gray01, mask, light=(-0.35, -0.35, 0.87), blur_px: float = 1.5
) -> float:
    """Korelasi Pearson (dalam mask) antara render Lambert relief dan kecerahan foto. Foto nyata juga dipengaruhi
    albedo dan arah cahaya yang tidak diketahui, jadi nilainya relatif: pakai untuk membandingkan varian."""
    r = ndi.gaussian_filter(render_lambert(h_mm, pixel_mm, light), blur_px)
    g = ndi.gaussian_filter(np.asarray(gray01, float), blur_px)
    m = np.asarray(mask, bool)
    if m.sum() < 10 or r[m].std() < 1e-9 or g[m].std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(r[m], g[m])[0, 1])


# --------------------------------------------------------------------------- penyaringan
def guided_filter(src, guide, radius: int = 8, eps: float = 1e-3):
    """Guided filter (He dkk.): haluskan `src` mengikuti tepi `guide` (gambar abu-abu 0..1)."""
    src = np.asarray(src, float)
    guide = np.asarray(guide, float)
    size = 2 * radius + 1

    def box(x):
        return ndi.uniform_filter(x, size=size, mode="nearest")

    mg, ms = box(guide), box(src)
    cov = box(guide * src) - mg * ms
    var = box(guide * guide) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return box(a) * guide + box(b)


# --------------------------------------------------------------------------- skala fisik dan resolusi
def sigma_mm_to_px(sigma_mm: float, mm_per_px: float) -> float:
    """Skala fusi/feather dalam mm lebih konsisten antar gambar daripada dalam piksel (mm_per_px berkisar 0,4-1,0)."""
    return sigma_mm / mm_per_px


def resample_heightmap(h_mm, src_pitch_mm: float, dst_pitch_mm: float, order: int = 1):
    """Ubah jarak piksel dari src ke dst (mm/px). Meningkatkan resolusi tidak menambah detail asli."""
    zoom = src_pitch_mm / dst_pitch_mm
    return ndi.zoom(np.asarray(h_mm, float), zoom, order=order, mode="nearest")


def camera_distance_mm(ipd_px: float, focal_px: float, ipd_mm: float = rc.DEFAULT_IPD_MM) -> float:
    """Jarak kamera ke bidang mata (mm): Z = f_px * IPD_mm / IPD_px. Jarak < ~500 mm = foto dekat, perspektif
    membesarkan hidung (depth model dan landmark sama-sama terdampak)."""
    return focal_px * ipd_mm / ipd_px


def focal_px_from_exif(focal_35mm: float, image_w_px: int, image_h_px: int) -> float:
    """Fokus dalam piksel dari fokus setara 35 mm (EXIF FocalLengthIn35mmFilm). Sensor 35 mm: lebar 36 mm.
    Perkiraan untuk gambar tak terpotong; hasil crop menggeser nilainya."""
    w_long = max(image_w_px, image_h_px)
    return focal_35mm * w_long / 36.0


# --------------------------------------------------------------------------- ansambel
def combine_aligned(maps, how: str = "median"):
    """Gabungkan beberapa peta tinggi yang SUDAH diselaraskan ke mm (mis. TTA flip/skala, atau beberapa model depth)."""
    a = np.stack([np.asarray(m, float) for m in maps])
    return np.median(a, 0) if how == "median" else a.mean(0)
