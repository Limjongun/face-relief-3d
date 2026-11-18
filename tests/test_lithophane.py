import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lithophane as lp
import relief_core as rc


def test_lithophane_pipeline(tmp_path):
    # Buat gambar dummy 50x50
    from PIL import Image

    arr = np.linspace(0, 255, 50 * 50, dtype=np.uint8).reshape(50, 50)
    img_p = tmp_path / "test.jpg"
    Image.fromarray(arr).save(img_p)

    out_d = tmp_path / "out_litho"
    report = lp.create_lithophane(
        image_path=str(img_p),
        out_dir=str(out_d),
        width_mm=50.0,
        max_grid_px=50,
        t_min_mm=0.8,
        t_max_mm=2.5,
        border_mm=1.0,
    )

    stl_file = out_d / "lithophane.stl"
    assert stl_file.exists()
    assert report["mesh"]["watertight"] is True
    assert report["mesh"]["volume_mm3"] > 0
