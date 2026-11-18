# Face Relief 3D — Petunjuk Eksekusi Manual

Pipeline konversi **Foto Wajah 2D → Bas-Relief & Lithophane STL Siap Cetak 3D**.

---

## 1. Struktur Folder

```
relief\
├── blueprint_face_relief_3dprint.md   # Dokumen spesifikasi & arsitektur lengkap
├── requirements.txt                   # Daftar dependensi Python
├── README.md                          # Panduan eksekusi ini
├── models/
│   ├── download_models.py             # Skrip download otomatis model Face Landmarker
│   └── face_landmarker.task           # Model MediaPipe (setelah diunduh)
├── data/
│   ├── create_sample_photo.py         # Skrip pembuat foto sampel sintetis
│   └── my_photos/
│       └── sample_face.jpg            # Contoh foto portrait
├── src/
│   ├── relief_core.py                 # Modul algoritma inti (matematika, fusi, STL)
│   ├── run_relief.py                  # CLI pipeline utama (MediaPipe + Depth Anything)
│   ├── lithophane.py                  # CLI pembuat varian Lithophane STL
│   └── wireframe.py                   # Modul varian patung wireframe 478 titik
├── tests/
│   ├── test_relief_core.py            # Unit test matematika inti (12 tes)
│   └── test_lithophane.py             # Unit test generator lithophane
└── out/                               # Output berkas STL & laporan metrik
```

---

## 2. Langkah-Langkah Eksekusi Manual

### Langkah 1: Install Dependensi yang Diperlukan

Lingkungan Python lokal Anda sudah memiliki **PyTorch (CUDA GPU)**, **Transformers**, **OpenCV**, dan **SciPy**.
Dependensi yang perlu diinstal adalah `mediapipe`:

```powershell
pip install mediapipe
```

*(Opsional, jika ingin memastikan seluruh dependensi terpenuhi):*
```powershell
pip install -r requirements.txt
```

---

### Langkah 2: Unduh Model Face Landmarker

Jalankan skrip pengunduh untuk mengambil model resmi MediaPipe (tersimpan di `models/face_landmarker.task`):

```powershell
python models/download_models.py
```

*Catatan: Model Depth Anything V2 (`depth-anything/Depth-Anything-V2-Small-hf`) akan otomatis di-cache oleh Hugging Face saat pertama kali `run_relief.py` dijalankan.*

---

### Langkah 3: Verifikasi Tes Inti

Pastikan seluruh 13 tes verifikasi matematika (IPD scale, IRLS affine alignment, fusi frekuensi, dan jaminan watertight mesh) berjalan lulus:

```powershell
pytest tests -v
```

---

### Langkah 4: Jalankan Pipeline Relief pada Foto Wajah

#### A. Menggunakan Foto Sendiri
Letakkan foto wajah frontal Anda di dalam folder `data/my_photos/` (misal: `data/my_photos/wajah_saya.jpg`), lalu jalankan:

```powershell
python src/run_relief.py data/my_photos/wajah_saya.jpg --out out/hasil_relief --target-mm 10 --gamma 0.8
```

#### Parameter yang Dapat Disesuaikan:
* `--target-mm 10.0` : Tinggi tonjolan relief di atas alas (standar FDM: 8–12 mm).
* `--gamma 0.8` : Kontras lengkungan wajah (< 1.0 mempertegas area pipi & dahi).
* `--ipd-mm 63.0` : Jarak antar pupil mata riil (standar 63 mm).
* `--base-mm 2.0` : Ketebalan plakat alas dasar (agar kaku saat dicetak).
* `--depth-model depth-anything/Depth-Anything-V2-Small-hf` : Pilihan model (Small / Base / Large).

#### B. Menguji Dry-Run (Tanpa Model AI / Mode Sintetis)
```powershell
python src/run_relief.py --synthetic --out out/test_synthetic
```

---

### Langkah 5: Menghasilkan Varian Lithophane

Untuk membuat plakat tembus cahaya (*lithophane*) dari foto:

```powershell
python src/lithophane.py data/my_photos/sample_face.jpg --out out/hasil_lithophane --width-mm 100 --t-min 0.8 --t-max 3.0
```

---

## 3. Panduan Cetak di 3D Slicer (Cura / PrusaSlicer / Bambu Studio)

### Untuk Relief Wajah (STL):
* **Orientasi**: Letakkan alas datar menempel pada build plate (relief menghadap ke atas).
* **Layer Height**: 0.12 mm – 0.16 mm (semakin tipis layer, semakin halus gradasi hidung dan mata).
* **Infill**: 15% – 20% (alas 2 mm akan tercetak solid).
* **Support**: **TIDAK PERLU** support karena tidak ada overhang > 45°.
* **Warna Filamen**: Putih, abu-abu muda, atau matte PLA.

### Untuk Lithophane (STL):
* **Orientasi**: Cetak **berdiri tegak** sejajar sumbu Y bed untuk kestabilan.
* **Infill**: 100% solid (wajib infill penuh agar transmisi cahaya seragam).
* **Warna Filamen**: Wajib PLA Putih (White).
