# Mini Photoshop Pro - Pengolahan Citra Digital

Aplikasi desktop pengolahan citra digital berbasis **Python + Tkinter**. Versi ini dibuat lebih rapi, fleksibel, dan mudah didemokan untuk proyek mata kuliah: fitur dikelompokkan di sidebar, parameter muncul dinamis, dan hasil edit dapat dilihat realtime pada panel **Before vs After**.

## Fitur Utama

### 1. UI Lebih Jelas
- Layout 3 kolom: **Kelompok Fitur**, **Preview Before/After**, dan **Parameter Fleksibel**.
- Toolbar cepat: Load, Save, Undo, Redo, Reset, Histogram, CNN.
- Status bar menampilkan file aktif, ukuran gambar, status proses, undo/redo.
- Panel bantuan fitur: setiap fitur punya deskripsi singkat.

### 2. Live Image Editing
- Centang **Live Preview** untuk melihat hasil realtime saat slider digeser.
- Klik **Apply** untuk menyimpan hasil preview ke state edit.
- Klik **Cancel** untuk membatalkan preview dan kembali ke state terakhir.
- Tersedia **Undo/Redo** bertingkat.

### 3. Parameter Fleksibel
- Slider dinamis sesuai fitur aktif.
- Preset: **Ringan**, **Sedang**, **Kuat**.
- Pilihan interpolasi transformasi: **nearest** atau **bilinear**.
- Kernel size, threshold, Canny low/high, sigma, iterations, quality JPEG, region K, hue, saturation, dan parameter lain dapat diatur manual.

### 4. Fitur Pengolahan Citra
- Image Management: load, save, reset, before-after preview.
- Basic: brightness/contrast, grayscale, resize/scaling.
- Enhancement: histogram equalization, sharpening fleksibel, smoothing.
- Geometric Transform: rotate, scale, translate, flip, crop drag area.
- Restoration: Gaussian blur, median filter, salt & pepper removal.
- Binary & Edge: threshold, Canny, Sobel, Prewitt, Robert, Laplacian, LoG, erosion, dilation.
- Color: RGB channel splitting, hue/saturation adjustment.
- Segmentation: threshold segmentation, edge-based segmentation, region-based K-Means.
- Compression: JPEG quality simulation, color quantization, RLE ratio estimation.
- Histogram Analysis: histogram before-after dengan deteksi channel otomatis: gambar RGB menampilkan R/G/B, gambar grayscale hanya menampilkan gray, lengkap dengan filter channel; saat fitur Channel Splitting RGB aktif, grafik otomatis mengikuti channel yang dipilih. Sekarang dilengkapi informasi **total piksel** dan label sumbu Y yang jelas.
- Machine Learning: CNN object recognition opsional memakai MobileNetV2.

## Instalasi

Disarankan memakai virtual environment.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

## Menjalankan Aplikasi

```bash
python main.pyw
```

Atau:

```bash
# Linux/Mac
./run.sh

# Windows
run.bat
```


## Cara Pakai Singkat

1. Klik **Load** dan pilih gambar JPG/PNG/BMP.
2. Pilih fitur dari sidebar kiri.
3. Atur slider/parameter pada panel kanan.
4. Jika **Live Preview** aktif, hasil langsung terlihat di panel After.
5. Klik **Apply** untuk menyimpan efek.
6. Gunakan **Undo/Redo** jika ingin kembali.
7. Klik **Histogram** untuk analisis distribusi intensitas.
8. Klik **Save** untuk menyimpan hasil.

## Crop Area

1. Drag area pada panel **After / Live Result**.
2. Klik tombol **Crop Area** pada panel kanan.
3. Hasil crop akan menjadi state edit baru dan bisa di-undo.

## Struktur File

```text
mini_photoshop_tkinter/
├── main.pyw
├── requirements.txt
├── requirements-ml.txt
├── run.bat
├── run.sh
├── FITUR_SESUAI_DOKUMEN.md
├── mini_photoshop/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── history.py
│   ├── image_processor.py
│   ├── ml.py
│   └── ui_components.py
└── tests/
    └── test_image_processor_smoke.py
```

## Verifikasi Dasar Developer

```bash
python -m py_compile main.pyw mini_photoshop/*.py tests/test_image_processor_smoke.py
python tests/test_image_processor_smoke.py
```

## Catatan

- GUI Tkinter membutuhkan environment desktop/display.
- Pada server/headless environment, smoke test algoritma tetap bisa dijalankan tanpa membuka GUI.
- Semua algoritma utama dipisah di `image_processor.py` agar mudah diuji dan dijelaskan saat presentasi.


## Versi Web FastAPI

Proyek ini juga sudah tersedia sebagai aplikasi web dengan backend FastAPI dan frontend HTML/CSS/JavaScript vanilla.

```bash
pip install -r requirements.txt
python run_web.py
```

Lalu buka `http://127.0.0.1:8000`. Detail lengkap ada di `README_WEB.md`.
