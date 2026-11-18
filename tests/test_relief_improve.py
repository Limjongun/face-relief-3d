import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import relief_core as rc  # noqa: E402
import relief_improve as ri  # noqa: E402


def test_zscale_invariance_of_affine_fit():
    """Koreksi penting: fit affine menyerap skala z. rmse_mm ikut berskala, tetapi rmse/rentang TIDAK berubah.
    Jadi menyapu z_scale tidak memperbaiki kecocokan; gunakan metrik ternormalisasi."""
    rng = np.random.default_rng(0)
    d = rng.uniform(0, 1, 300)
    h = 20 * d + 2 + rng.normal(0, 1.0, 300)
    base = rc.fit_affine_robust(d, h)
    n0 = ri.normalized_rmse(base["rmse_mm"], h)
    for s in (0.5, 2.0):
        f = rc.fit_affine_robust(d, s * h)
        assert f["rmse_mm"] == pytest.approx(s * base["rmse_mm"], rel=1e-6)
        assert ri.normalized_rmse(f["rmse_mm"], s * h) == pytest.approx(n0, rel=1e-6)


def test_poly_beats_affine_on_curved_data_and_is_robust():
    rng = np.random.default_rng(1)
    d = rng.uniform(0, 1, 400)
    h = 30 * d**2 + 5 * d + rng.normal(0, 0.3, 400)
    bad = rng.choice(400, 40, replace=False)
    h[bad] += 25
    f1 = ri.fit_poly_robust(d, h, 1)
    f2 = ri.fit_poly_robust(d, h, 2)
    assert f2["rmse_mm"] < 0.6 * f1["rmse_mm"]
    assert f2["coef"][0] == pytest.approx(30, rel=0.05)
    assert 0.8 < f2["inlier_frac"] < 0.97
    # deg=1 harus setara fit_affine_robust
    ref = rc.fit_affine_robust(d, h)
    assert f1["rmse_mm"] == pytest.approx(ref["rmse_mm"], rel=1e-6)


def test_disparity_model_beats_affine_when_relation_is_inverse():
    rng = np.random.default_rng(2)
    true_h = rng.uniform(-20, 20, 500)  # tinggi sebenarnya (mm)
    z0 = 60.0
    d = 1.0 / (z0 - true_h) * 100  # disparitas ~ 1/kedalaman
    d_noisy = d + rng.normal(0, 0.002, 500)
    aff = rc.fit_affine_robust(d_noisy, true_h)
    dsp = ri.fit_disparity_model(d_noisy, true_h)
    assert dsp["rmse_mm"] < aff["rmse_mm"]
    pred = ri.apply_disparity_model(dsp, d_noisy)
    assert np.sqrt(np.mean((pred - true_h) ** 2)) < 1.5


def test_residuals_by_ring_locates_error():
    rng = np.random.default_rng(3)
    pts = rng.uniform(-100, 100, (800, 2))
    center = np.array([0.0, 0.0])
    dist = np.linalg.norm(pts, axis=1) / np.linalg.norm(pts, axis=1).max()
    resid = np.where(dist > 0.75, rng.normal(0, 8, 800), rng.normal(0, 1, 800))
    out = ri.residuals_by_ring(pts, resid, center)
    assert out["tepi"][1] > 4 * out["inti"][1]
    assert sum(v[0] for v in out.values()) == 800


def _face_like(h=100, w=100):
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cx, cy = (w - 1) / 2, h / 2
    return 20 * np.exp(-(((xx - cx) / 20) ** 2 + ((yy - cy) / 25) ** 2))


def test_mirror_symmetry():
    h = _face_like()
    mask = h > 0.5
    axis = 49.5
    assert ri.mirror_symmetry_error(h, mask, axis) < 1e-6
    asym = h.copy()
    asym[:, :50] += 3.0
    assert ri.mirror_symmetry_error(asym, mask, axis) > 1.0


def test_shading_consistency_prefers_correct_shape():
    h = _face_like()
    pitch = 0.5
    gray = ri.render_lambert(h, pitch)
    mask = h > 0.3
    good = ri.shading_consistency(h, pitch, gray, mask)
    wrong = ri.shading_consistency(np.flipud(h) * 0 + _face_like()[:, ::-1].T[:100, :100] * 0.2, pitch, gray, mask)
    flat = ri.shading_consistency(np.zeros_like(h), pitch, gray, mask)
    assert good > 0.95
    assert np.isnan(flat) or flat < good


def test_guided_filter_denoises_but_keeps_edge():
    rng = np.random.default_rng(4)
    img = np.zeros((80, 80))
    img[:, 40:] = 10.0  # tepi vertikal
    noisy = img + rng.normal(0, 1.0, img.shape)
    out = ri.guided_filter(noisy, img / 10.0, radius=5, eps=1e-3)
    flat_region = (slice(10, 70), slice(5, 30))
    assert out[flat_region].std() < 0.5 * noisy[flat_region].std()
    step = out[:, 41:44].mean() - out[:, 36:39].mean()
    assert step > 8.5  # tepi tetap tajam
    # filter biasa (uniform) mengaburkan tepi lebih parah
    blurred = ndi_uniform(noisy, 11)
    assert (blurred[:, 41:44].mean() - blurred[:, 36:39].mean()) < step


def ndi_uniform(x, k):
    from scipy import ndimage as ndi

    return ndi.uniform_filter(x, k)


def test_sigma_and_resample_and_distance():
    assert ri.sigma_mm_to_px(2.4, 0.4) == pytest.approx(6.0)
    assert ri.sigma_mm_to_px(2.4, 1.0) == pytest.approx(2.4)
    h = _face_like(60, 80)
    out = ri.resample_heightmap(h, 1.0, 0.5)
    assert out.shape == (120, 160)
    assert out.max() == pytest.approx(h.max(), rel=0.02)
    # IPD 63 mm terlihat 120 px dengan fokus 1500 px -> jarak ~ 787 mm
    assert ri.camera_distance_mm(120, 1500) == pytest.approx(787.5)
    assert ri.focal_px_from_exif(26.0, 4000, 3000) == pytest.approx(26.0 * 4000 / 36.0)


def test_combine_aligned():
    a, b, c = np.full((5, 5), 1.0), np.full((5, 5), 2.0), np.full((5, 5), 100.0)
    assert ri.combine_aligned([a, b, c], "median")[0, 0] == 2.0
    assert ri.combine_aligned([a, b, c], "mean")[0, 0] == pytest.approx(103 / 3)


def test_regional_weights_and_weighted_affine():
    pts = np.array([[0, 0], [40, 0], [80, 0]], float)
    w = ri.regional_weights(pts, center_xy=[0, 0], edges=(0.3, 0.7), weights=(1.0, 0.8, 0.25))
    assert w[0] == 1.0 and w[1] == 0.8 and w[2] == 0.25
    rng = np.random.default_rng(5)
    d = rng.uniform(0, 1, 100)
    h = 20 * d + 2
    fit = ri.fit_affine_weighted_robust(d, h, prior_weights=np.ones_like(d))
    assert fit["a"] == pytest.approx(20, rel=1e-3)
    assert fit["b"] == pytest.approx(2, rel=1e-3)
