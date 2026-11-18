"""Skrip Pengunduh Model Face Landmarker MediaPipe.

Menyimpan model ke direktori models/face_landmarker.task
"""
import os
import sys
import urllib.request
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DEST_DIR = Path(__file__).resolve().parent
DEST_FILE = DEST_DIR / "face_landmarker.task"


def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_tot = total_size / (1024 * 1024)
        sys.stdout.write(f"\rUnduh: {percent:5.1f}% [{mb_down:.1f} MB / {mb_tot:.1f} MB]")
        sys.stdout.flush()


def download_model(force: bool = False):
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    if DEST_FILE.exists() and not force:
        size_mb = DEST_FILE.stat().st_size / (1024 * 1024)
        print(f"[✓] Model sudah ada: {DEST_FILE} ({size_mb:.2f} MB)")
        print("Gunakan flag --force jika ingin mengunduh ulang.")
        return

    print(f"[*] Mengunduh model MediaPipe Face Landmarker dari:")
    print(f"    {MODEL_URL}")
    print(f"[*] Menyimpan ke: {DEST_FILE}")

    try:
        urllib.request.urlretrieve(MODEL_URL, DEST_FILE, reporthook=reporthook)
        print("\n[✓] Pengunduhan berhasil selesai!")
    except Exception as e:
        if DEST_FILE.exists():
            DEST_FILE.unlink()
        print(f"\n[X] Gagal mengunduh model: {e}")
        print("Alternatif: Anda dapat mengunduh file secara manual via browser lalu simpan di folder 'models/face_landmarker.task'")
        sys.exit(1)


if __name__ == "__main__":
    force_download = "--force" in sys.argv
    download_model(force=force_download)
