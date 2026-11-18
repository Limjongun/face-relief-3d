# Ringkasan Eksperimen A1 — Penyelarasan Nonlinear

## 1. Rata-rata nRMSE (Normalized RMSE)
* **Baseline (Affine)**: `0.1086`
* **Varian 1 (Polinomial deg=2)**: `0.1083` (Lebih Baik)
* **Varian 2 (Disparitas h=a/(d+s)+b)**: `0.1084` (Lebih Baik)

## 2. Kriteria Penerimaan (Diterima jika nRMSE turun pada ≥ 80% foto)
* Polinomial mengalahkan Affine pada: **7/21 foto (33.3%)**
* Model Disparitas mengalahkan Affine pada: **7/21 foto (33.3%)**

## 3. Analisis Residu Cincin (Inti vs Tepi)
* Rata-rata galat inti hidung/mata: `3.79 mm`
* Rata-rata galat tepi rahang: `13.11 mm`
