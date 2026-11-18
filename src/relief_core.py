"""Inti pipeline relief wajah: landmark + peta depth -> heightmap -> STL padat (watertight).

Hanya butuh numpy, scipy, Pillow. Tidak memerlukan mediapipe/torch, sehingga bisa diuji terpisah.

Konvensi
--------
* Peta depth dari Depth Anything adalah *disparity relatif*: nilai BESAR = DEKAT kamera.
* Landmark MediaPipe: x,y ternormalisasi [0,1], z relatif (skalanya mirip x), z NEGATIF = lebih dekat kamera.
* Tinggi relief h (mm): makin dekat kamera makin tinggi. Jadi h_landmark = -z * lebar_gambar_px * mm_per_px.
"""
from __future__ import annotations

import struct

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

IRIS_CENTER_A = 468  # pusat iris (landmark ke-468 pada 478 titik, refine/iris)
IRIS_CENTER_B = 473
DEFAULT_IPD_MM = 63.0  # rata-rata jarak antar pupil orang dewasa (bervariasi ~54-74 mm)


# --------------------------------------------------------------------------- skala dan landmark
def mm_per_pixel_from_ipd(iris_a_xy, iris_b_xy, ipd_mm: float = DEFAULT_IPD_MM) -> float:
    """Skala mm/piksel dari jarak antar dua pusat iris (dalam piksel)."""
    d = float(np.hypot(*(np.asarray(iris_a_xy, float) - np.asarray(iris_b_xy, float))))
    if d <= 1e-6:
        raise ValueError("Jarak iris nol: landmark iris tidak valid.")
    return ipd_mm / d


def landmarks_to_px_and_height(lm_xyz_norm, img_w: int, img_h: int, mm_per_px: float):
    """lm_xyz_norm: (N,3) x,y,z ala MediaPipe. Return (pts_xy_px (N,2), tinggi_mm (N,)).
    z MediaPipe berskala sama dengan x => z_px = z * img_w. Tinggi = -z (dekat = tinggi)."""
    lm = np.asarray(lm_xyz_norm, float)
    pts = np.stack([lm[:, 0] * img_w, lm[:, 1] * img_h], axis=1)
    h_mm = -lm[:, 2] * img_w * mm_per_px
    return pts, h_mm


# --------------------------------------------------------------------------- penyelarasan depth
def sample_map(arr, pts_xy):
    """Sampel bilinear arr[y, x] pada titik (x,y) piksel. Titik di luar dijepit ke tepi."""
    h, w = arr.shape
    x = np.clip(pts_xy[:, 0], 0, w - 1)
    y = np.clip(pts_xy[:, 1], 0, h - 1)
    return ndi.map_coordinates(arr, [y, x], order=1, mode="nearest")


def fit_affine_robust(d, h, n_iter: int = 30, c: float = 4.685):
    """Cocokkan h ~ a*d + b dengan IRLS (bobot Tukey bisquare, skala MAD).
    d = nilai depth di titik landmark, h = tinggi landmark (mm).
    Return dict(a, b, rmse_mm, inlier_frac)."""
    d = np.asarray(d, float)
    h = np.asarray(h, float)
    A = np.stack([d, np.ones_like(d)], axis=1)
    w = np.ones_like(d)
    coef = np.linalg.lstsq(A, h, rcond=None)[0]
    for _ in range(n_iter):
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(A * sw[:, None], h * sw, rcond=None)[0]
        r = h - A @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = r / (c * s)
        w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        if w.sum() < 3:
            break
    r = h - A @ coef
    inl = w > 0
    return dict(
        a=float(coef[0]),
        b=float(coef[1]),
        rmse_mm=float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else float("nan"),
        inlier_frac=float(inl.mean()),
    )


# --------------------------------------------------------------------------- mask
def polygon_mask(shape, poly_xy) -> np.ndarray:
    """Mask boolean (H,W) dari poligon (N,2) piksel."""
    h, w = shape
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).polygon([tuple(map(float, p)) for p in poly_xy], fill=1)
    return np.array(im, dtype=bool)


def convex_hull_polygon(pts_xy) -> np.ndarray:
    from scipy.spatial import ConvexHull

    pts = np.asarray(pts_xy, float)
    return pts[ConvexHull(pts).vertices]


def soft_mask(mask, feather_px: float) -> np.ndarray:
    """Alpha 0..1: 1 di dalam, turun halus (smoothstep) ke 0 pada jarak feather_px dari tepi dalam."""
    if feather_px <= 0:
        return mask.astype(float)
    dist = ndi.distance_transform_edt(mask)
    t = np.clip(dist / feather_px, 0, 1)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------- bentuk kasar + fusi detail
def lowfreq_from_landmarks(pts_xy, h_mm, shape, sigma_px: float = 6.0) -> np.ndarray:
    """Permukaan halus dari tinggi landmark (interpolasi Delaunay, luar hull = tetangga terdekat) lalu Gaussian."""
    hgt, wid = shape
    yy, xx = np.mgrid[0:hgt, 0:wid]
    lin = LinearNDInterpolator(pts_xy, h_mm)
    near = NearestNDInterpolator(pts_xy, h_mm)
    g = lin(xx, yy)
    g = np.where(np.isnan(g), near(xx, yy), g)
    return ndi.gaussian_filter(g, sigma_px) if sigma_px > 0 else g


def fuse_frequencies(low, depth_aligned, sigma_px: float = 6.0, w_detail: float = 1.0) -> np.ndarray:
    """fusi = frekuensi RENDAH dari landmark + w * frekuensi TINGGI dari depth.
    Frekuensi tinggi = depth_aligned - Gaussian(depth_aligned, sigma_px)."""
    hp = depth_aligned - ndi.gaussian_filter(depth_aligned, sigma_px)
    return low + w_detail * hp


# --------------------------------------------------------------------------- kompres relief
def compress_relief(h, alpha, target_mm: float = 10.0, gamma: float = 0.8, p_low: float = 1, p_high: float = 99):
    """Normalisasi tinggi di dalam wajah ke [0,1] (persentil p_low..p_high), pangkat gamma, skala ke target_mm.
    Di luar wajah (alpha=0) tinggi = 0. Return peta tinggi (mm) di [0, target_mm]."""
    inside = alpha > 0.5
    lo, hi = np.percentile(h[inside], [p_low, p_high])
    if hi - lo < 1e-9:
        raise ValueError("Relief datar (rentang tinggi ~0).")
    n = np.clip((h - lo) / (hi - lo), 0, 1) ** gamma
    return target_mm * n * alpha


# --------------------------------------------------------------------------- mesh padat
def heightmap_to_solid(h_mm, pixel_mm: float, base_mm: float = 2.0) -> np.ndarray:
    """Heightmap (H,W) mm -> segitiga (T,3,3) mesh PADAT watertight.
    Atas: permukaan relief (z = base + h). Bawah: bidang datar z=0. Sisi: dinding di 4 tepi.
    Sumbu: x ke kanan, y ke ATAS gambar, z ke atas. Winding diperbaiki agar volume positif."""
    h_mm = np.asarray(h_mm, float)
    H, W = h_mm.shape
    if H < 2 or W < 2:
        raise ValueError("Heightmap minimal 2x2.")
    ys, xs = np.mgrid[0:H, 0:W]
    X = xs * pixel_mm
    Y = (H - 1 - ys) * pixel_mm
    top = np.stack([X, Y, h_mm + base_mm], axis=-1).reshape(-1, 3)
    bot = np.stack([X, Y, np.zeros_like(X)], axis=-1).reshape(-1, 3)
    n = H * W
    V = np.vstack([top, bot])  # indeks bawah = indeks atas + n

    idx = np.arange(n).reshape(H, W)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]  # sel: a b / c d
    F = []
    # atas
    F.append(np.stack([a, c, b], -1).reshape(-1, 3))
    F.append(np.stack([b, c, d], -1).reshape(-1, 3))
    # bawah (dibalik)
    F.append(np.stack([a + n, b + n, c + n], -1).reshape(-1, 3))
    F.append(np.stack([b + n, d + n, c + n], -1).reshape(-1, 3))

    def wall(i0, i1):  # sepanjang tepi i0->i1 (urutan sepanjang batas)
        t0, t1 = i0, i1
        b0, b1 = i0 + n, i1 + n
        return [np.stack([t0, t1, b0], -1), np.stack([t1, b1, b0], -1)]

    top_row, bot_row = idx[0, :], idx[-1, :]
    left_col, right_col = idx[:, 0], idx[:, -1]
    walls = []
    walls += wall(top_row[:-1], top_row[1:])
    walls += wall(bot_row[1:], bot_row[:-1])
    walls += wall(left_col[1:], left_col[:-1])
    walls += wall(right_col[:-1], right_col[1:])
    F += walls
    F = np.vstack(F)
    tris = V[F]
    if signed_volume(tris) < 0:
        tris = tris[:, [0, 2, 1], :]
    return tris


# --------------------------------------------------------------------------- cek mesh + STL
def signed_volume(tris) -> float:
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->", v0, np.cross(v1, v2)) / 6.0)


def check_mesh(tris, decimals: int = 6) -> dict:
    """Cek watertight: tiap sisi terarah muncul tepat 1x dan kebalikannya tepat 1x (manifold + orientasi konsisten)."""
    tris = np.asarray(tris, float)
    key = np.round(tris.reshape(-1, 3), decimals)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1, 3)
    degenerate = int(((inv[:, 0] == inv[:, 1]) | (inv[:, 1] == inv[:, 2]) | (inv[:, 0] == inv[:, 2])).sum())
    e = np.concatenate([inv[:, [0, 1]], inv[:, [1, 2]], inv[:, [2, 0]]])
    m = int(inv.max()) + 1
    fwd = e[:, 0].astype(np.int64) * m + e[:, 1]
    rev = e[:, 1].astype(np.int64) * m + e[:, 0]
    fu, fc = np.unique(fwd, return_counts=True)
    ok_unique = bool((fc == 1).all())
    ok_pair = bool(np.isin(fu, rev).all()) and len(fu) == len(np.unique(rev))
    return dict(
        n_triangles=int(len(tris)),
        watertight=bool(ok_unique and ok_pair and degenerate == 0),
        degenerate_triangles=degenerate,
        volume_mm3=signed_volume(tris),
        bbox_mm=(tris.reshape(-1, 3).max(0) - tris.reshape(-1, 3).min(0)).round(3).tolist(),
    )


def write_binary_stl(path, tris, header: str = "face-relief") -> None:
    tris = np.asarray(tris, np.float32)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = n, tris
    with open(path, "wb") as f:
        f.write(header.encode("ascii", "ignore")[:80].ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        f.write(rec.tobytes())


def read_binary_stl(path) -> np.ndarray:
    with open(path, "rb") as f:
        f.seek(80)
        n = struct.unpack("<I", f.read(4))[0]
        rec = np.frombuffer(
            f.read(), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")], count=n
        )
    return rec["v"].astype(float)


# --------------------------------------------------------------------------- lithophane (varian)
def lithophane_thickness(gray01, t_min_mm: float = 0.8, t_max_mm: float = 3.0, gamma: float = 1.0):
    """Ketebalan dari kecerahan: terang -> tipis (cahaya tembus), gelap -> tebal. gray01 di [0,1]."""
    g = np.clip(np.asarray(gray01, float), 0, 1)
    return t_min_mm + (t_max_mm - t_min_mm) * (1 - g) ** gamma
