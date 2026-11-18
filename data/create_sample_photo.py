"""Membuat citra wajah sintetis (sample portrait) untuk pengujian awal."""
from pathlib import Path

import cv2
import numpy as np


def generate_portrait(output_path: str = "data/my_photos/sample_face.jpg"):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    w, h = 600, 800
    img = np.full((h, w, 3), 220, dtype=np.uint8)  # Background terang

    # Bentuk kepala
    cx, cy = w // 2, h // 2
    cv2.ellipse(img, (cx, cy), (160, 220), 0, 0, 360, (190, 190, 190), -1)

    # Mata (pusat iris kira-kira di x=240 dan x=360, jarak 120 px)
    eye_y = cy - 40
    cv2.circle(img, (cx - 60, eye_y), 15, (60, 60, 60), -1)
    cv2.circle(img, (cx + 60, eye_y), 15, (60, 60, 60), -1)
    # Pupil
    cv2.circle(img, (cx - 60, eye_y), 6, (20, 20, 20), -1)
    cv2.circle(img, (cx + 60, eye_y), 6, (20, 20, 20), -1)

    # Hidung (shading)
    nose_pts = np.array([[cx, eye_y + 20], [cx - 15, eye_y + 80], [cx + 15, eye_y + 80]], np.int32)
    cv2.fillPoly(img, [nose_pts], (150, 150, 150))

    # Mulut
    mouth_y = eye_y + 130
    cv2.ellipse(img, (cx, mouth_y), (40, 15), 0, 0, 180, (120, 120, 120), -1)

    # Haluskan dengan Gaussian blur agar gradasi menyerupai foto
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    cv2.imwrite(str(out), blurred)
    print(f"[✓] Citra sampel portrait berhasil dibuat: {out}")


if __name__ == "__main__":
    generate_portrait()
