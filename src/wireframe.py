"""Generator Patung Wireframe Landmark Wajah 3D (Varian Kreatif).

Menghubungkan 478 titik landmark MediaPipe menjadi bola-bola kecil dan
batang-batang silinder struktural siap cetak 3D (disarankan SLA/Resin).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import relief_core as rc
except ImportError:
    from src import relief_core as rc


def create_sphere(center: np.ndarray, radius: float = 1.0, subdivisions: int = 2) -> np.ndarray:
    """Membuat mesh bola (icosphere atau UV sphere sederhana) pada titik center."""
    u = np.linspace(0, 2 * np.pi, 8)
    v = np.linspace(0, np.pi, 6)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]

    tris = []
    for i in range(len(u) - 1):
        for j in range(len(v) - 1):
            p0 = [x[i, j], y[i, j], z[i, j]]
            p1 = [x[i + 1, j], y[i + 1, j], z[i + 1, j]]
            p2 = [x[i, j + 1], y[i, j + 1], z[i, j + 1]]
            p3 = [x[i + 1, j + 1], y[i + 1, j + 1], z[i + 1, j + 1]]
            tris.append([p0, p1, p2])
            tris.append([p1, p3, p2])
    return np.array(tris, dtype=np.float32)


def create_cylinder(p1: np.ndarray, p2: np.ndarray, radius: float = 0.6, segments: int = 6) -> np.ndarray:
    """Membuat silinder antara titik p1 dan p2."""
    v = p2 - p1
    length = np.linalg.norm(v)
    if length < 1e-6:
        return np.zeros((0, 3, 3), dtype=np.float32)

    v_norm = v / length
    # Cari vektor ortogonal
    not_v = np.array([1, 0, 0]) if abs(v_norm[0]) < 0.8 else np.array([0, 1, 0])
    n1 = np.cross(v_norm, not_v)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v_norm, n1)

    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    circle = np.array([np.cos(a) * n1 + np.sin(a) * n2 for a in angles]) * radius

    ring1 = p1 + circle
    ring2 = p2 + circle

    tris = []
    for i in range(segments):
        next_i = (i + 1) % segments
        # 2 segitiga untuk dinding selimut silinder
        tris.append([ring1[i], ring2[i], ring1[next_i]])
        tris.append([ring1[next_i], ring2[i], ring2[next_i]])

    return np.array(tris, dtype=np.float32)


def generate_wireframe_mesh(
    landmarks_xyz_mm: np.ndarray,
    connections: list[tuple[int, int]] | None = None,
    sphere_radius_mm: float = 0.9,
    strut_radius_mm: float = 0.5,
) -> np.ndarray:
    """Menggabungkan landmark titik menjadi jaring struktur STL wireframe."""
    mesh_tris = []
    n_pts = len(landmarks_xyz_mm)

    print(f"[*] Membuat {n_pts} bola sendi landmark...")
    for pt in landmarks_xyz_mm:
        mesh_tris.append(create_sphere(pt, radius=sphere_radius_mm))

    if connections:
        print(f"[*] Membuat {len(connections)} batang strut penghubung...")
        for i, j in connections:
            if i < n_pts and j < n_pts:
                cyl = create_cylinder(landmarks_xyz_mm[i], landmarks_xyz_mm[j], radius=strut_radius_mm)
                if len(cyl) > 0:
                    mesh_tris.append(cyl)

    return np.vstack(mesh_tris)
