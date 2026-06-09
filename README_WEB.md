# Mini Photoshop Web - FastAPI Backend + Frontend

Versi ini memigrasikan proyek Tkinter menjadi aplikasi web:

- **Backend**: FastAPI (`backend/main.py`) memakai ulang algoritma di `mini_photoshop/image_processor.py`.
- **Frontend**: HTML/CSS/JavaScript vanilla (`frontend/`) tanpa build step Node.js.
- **Fitur web**: load image, before/after preview, live preview, apply/cancel, undo/redo, reset, crop drag area, histogram (dengan total piksel dan label Y), export custom filename (PNG/JPEG/BMP/TIFF), dan endpoint CNN opsional.

## Cara Menjalankan

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
python run_web.py
```

Buka browser ke:

```text
http://127.0.0.1:8000
```

Alternatif menjalankan langsung dengan Uvicorn:

```bash
uvicorn backend.main:app --reload
```

## Endpoint Utama

- `GET /` - membuka frontend web.
- `GET /api/features` - daftar fitur, kontrol, preset, dan model CNN.
- `POST /api/process` - memproses gambar berdasarkan fitur dan parameter.
- `POST /api/export` - konversi/download hasil ke PNG, JPEG, BMP, atau TIFF.
- `POST /api/histogram` - mengambil histogram grayscale/RGB.
- `POST /api/cnn` - klasifikasi CNN opsional jika TensorFlow terinstal.

## Catatan CNN

CNN tetap opsional seperti proyek asli. Untuk mengaktifkannya:

```bash
pip install -r requirements-ml.txt
```

TensorFlow/Keras dapat mengunduh bobot ImageNet saat model pertama kali digunakan.

## Verifikasi Developer

```bash
python -m py_compile backend/main.py run_web.py mini_photoshop/*.py tests/test_image_processor_smoke.py
python tests/test_image_processor_smoke.py
```
