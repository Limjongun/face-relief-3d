import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import relief_core as rc  # noqa: E402


def synthetic_face(h=120, w=100):
    """Kepala elips + hidung (bukit Gaussian). Return (tinggi_mm_penuh, titik_landmark_px, tinggi_landmark)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cx, cy = w / 2, h / 2
    ell = ((xx - cx) / (w * 0.4)) ** 2 + ((yy - cy) / (h * 0.45)) ** 2
    head = np.clip(1 - ell, 0, None) ** 0.5 * 30
    nose = 12 * np.exp(-(((xx - cx) / 6) ** 2 + ((yy - cy - 8) / 10) ** 2))
    full = head + nose
    rng = np.random.default_rng(0)
    pts = np.stack([rng.uniform(w * 0.15, w * 0.85, 300), rng.uniform(h * 0.1, h * 0.9, 300)], 1)
    hl = rc.sample_map(full, pts)
    return full, pts, hl


def test_mm_per_pixel_from_ipd():
    assert rc.mm_per_pixel_from_ipd((100, 100), (163, 100), 63.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rc.mm_per_pixel_from_ipd((1, 1), (1, 1))


def test_landmark_height_sign_and_scale():
    lm = np.array([[0.5, 0.5, -0.05], [0.4, 0.4, 0.02]])
    pts, h = rc.landmarks_to_px_and_height(lm, 200, 100, 0.5)
    assert pts[0].tolist() == [100.0, 50.0]
    assert h[0] == pytest.approx(0.05 * 200 * 0.5)  # z negatif -> tinggi positif
    assert h[1] < 0


def test_affine_recovery_with_outliers():
    rng = np.random.default_rng(1)
    d = rng.uniform(0, 1, 400)
    h = 25 * d + 3 + rng.normal(0, 0.2, 400)
    bad = rng.choice(400, 60, replace=False)
    h[bad] += rng.uniform(15, 40, 60)
    fit = rc.fit_affine_robust(d, h)
    assert fit["a"] == pytest.approx(25, rel=0.03)
    assert fit["b"] == pytest.approx(3, abs=0.5)
    assert fit["rmse_mm"] < 0.5
    assert 0.7 < fit["inlier_frac"] < 0.95
    # OLS biasa harus lebih buruk -> membuktikan IRLS berguna
    a_ols = np.polyfit(d, h, 1)[0]
    assert abs(a_ols - 25) > abs(fit["a"] - 25)


def test_depth_alignment_recovers_shape():
    full, pts, hl = synthetic_face()
    depth = (full - 7.0) / 30.0  # affine tak dikenal (disparity relatif)
    d_at = rc.sample_map(depth, pts)
    fit = rc.fit_affine_robust(d_at, hl)
    aligned = fit["a"] * depth + fit["b"]
    assert np.abs(aligned - full).max() < 0.5


def test_mask_and_feather():
    poly = [(10, 10), (90, 10), (90, 90), (10, 90)]
    m = rc.polygon_mask((100, 100), poly)
    assert m[50, 50] and not m[2, 2]
    a = rc.soft_mask(m, 8)
    assert a[50, 50] == pytest.approx(1.0)
    assert a[2, 2] == 0
    assert 0 < a[12, 50] < 1
    assert np.array_equal(rc.soft_mask(m, 0), m.astype(float))


def test_convex_hull():
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], float)
    hull = rc.convex_hull_polygon(pts)
    assert len(hull) == 4


def test_lowfreq_and_fuse():
    full, pts, hl = synthetic_face()
    low = rc.lowfreq_from_landmarks(pts, hl, full.shape, sigma_px=4)
    assert low.shape == full.shape and np.isfinite(low).all()
    # bentuk kasar mendekati kebenaran di dalam hull
    inner = full[40:80, 30:70]
    assert np.abs(low[40:80, 30:70] - inner).mean() < 3.0
    # detail frekuensi tinggi: tambahkan tekstur ke depth
    rng = np.random.default_rng(2)
    tex = ndi_smooth_noise(full.shape, rng)
    f_tex = rc.fuse_frequencies(low, full + tex, sigma_px=4, w_detail=1.0)
    f_plain = rc.fuse_frequencies(low, full, sigma_px=4, w_detail=1.0)
    delta = f_tex - f_plain  # kontribusi tekstur = hp(tex)
    from scipy import ndimage as ndi
    hp_tex = tex - ndi.gaussian_filter(tex, 4)
    assert np.allclose(delta, hp_tex, atol=1e-9)
    assert delta.std() > 0.1
    # w=0 -> tepat sama dengan low
    assert np.allclose(rc.fuse_frequencies(low, full + tex, 4, 0.0), low)


def ndi_smooth_noise(shape, rng):
    from scipy import ndimage as ndi

    n = ndi.gaussian_filter(rng.normal(size=shape), 1.0)
    return n / n.std() * 0.3


def test_compress_relief():
    full, _, _ = synthetic_face()
    poly = [(15, 12), (85, 12), (85, 108), (15, 108)]
    a = rc.soft_mask(rc.polygon_mask(full.shape, poly), 6)
    out = rc.compress_relief(full + 5, a, target_mm=10, gamma=0.8)
    assert out.max() == pytest.approx(10, abs=0.5)
    assert out.min() >= 0
    assert out[0, 0] == 0  # luar mask = 0
    with pytest.raises(ValueError):
        rc.compress_relief(np.ones_like(full), a)


def test_solid_mesh_watertight_and_volume(tmp_path):
    rng = np.random.default_rng(3)
    h = rng.uniform(0, 5, (20, 30))
    tris = rc.heightmap_to_solid(h, pixel_mm=0.5, base_mm=2.0)
    info = rc.check_mesh(tris)
    assert info["watertight"], info
    assert info["degenerate_triangles"] == 0
    assert info["volume_mm3"] > 0
    # jumlah segitiga: atas+bawah 2*2*(H-1)(W-1) + dinding 2*2*((W-1)+(H-1))
    H, W = h.shape
    assert info["n_triangles"] == 4 * (H - 1) * (W - 1) + 4 * ((W - 1) + (H - 1))
    # volume analitik kasar: integral trapesium sel = rata-rata 4 sudut * luas sel
    cell = 0.5 * 0.5
    z = h + 2.0
    za, zb, zc, zd = z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]
    # dua segitiga per sel: (a,c,b) dan (b,c,d); volume prisma = luas * rata-rata tinggi 3 titik
    vol = (cell / 2 * ((za + zc + zb) / 3 + (zb + zc + zd) / 3)).sum()
    assert info["volume_mm3"] == pytest.approx(vol, rel=1e-6)
    # bbox
    assert info["bbox_mm"][0] == pytest.approx((W - 1) * 0.5)
    # STL round trip
    p = tmp_path / "x.stl"
    rc.write_binary_stl(p, tris)
    assert os.path.getsize(p) == 84 + 50 * len(tris)
    back = rc.read_binary_stl(p)
    assert back.shape == tris.shape
    assert np.allclose(back, tris, atol=1e-4)
    assert rc.check_mesh(back, decimals=4)["watertight"]


def test_check_mesh_detects_hole():
    tris = rc.heightmap_to_solid(np.ones((5, 5)), 1.0)
    holed = tris[:-1]  # buang 1 segitiga
    assert not rc.check_mesh(holed)["watertight"]


def test_end_to_end(tmp_path):
    full, pts, hl = synthetic_face()
    depth = (full - 7.0) / 30.0
    fit = rc.fit_affine_robust(rc.sample_map(depth, pts), hl)
    aligned = fit["a"] * depth + fit["b"]
    low = rc.lowfreq_from_landmarks(pts, hl, full.shape, 5)
    fused = rc.fuse_frequencies(low, aligned, 5, 1.0)
    hull = rc.convex_hull_polygon(pts)
    alpha = rc.soft_mask(rc.polygon_mask(full.shape, hull), 5)
    relief = rc.compress_relief(fused, alpha, 10, 0.8)
    tris = rc.heightmap_to_solid(relief, 0.5, 2.0)
    info = rc.check_mesh(tris)
    assert info["watertight"] and info["volume_mm3"] > 0
    rc.write_binary_stl(tmp_path / "face.stl", tris)


def test_lithophane():
    t = rc.lithophane_thickness(np.array([0.0, 0.5, 1.0]), 0.8, 3.0, 1.0)
    assert t[0] == pytest.approx(3.0) and t[2] == pytest.approx(0.8)
    assert t[1] == pytest.approx(1.9)
