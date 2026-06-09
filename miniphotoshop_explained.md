# 📸 Penjelasan Komprehensif: Project Mini Photoshop — Pengolahan Citra Digital

> Dokumen ini menjelaskan **struktur project**, **arsitektur kode**, dan **hubungan setiap fungsi dengan rumus/teori asli** mata kuliah Pengolahan Citra Digital.

---

## 1. Gambaran Umum Project

Mini Photoshop ini adalah aplikasi pengolahan citra digital dengan **dua mode deployment**:

| Mode | Stack | File Utama |
|---|---|---|
| Desktop GUI | Python + Tkinter | `mini_photoshop/app.py` |
| Web App | FastAPI (backend) + HTML/JS vanilla (frontend) | `backend/main.py` + `frontend/app.js` |

Kedua mode **berbagi satu inti algoritma** yang sama, yaitu `mini_photoshop/image_processor.py`. Ini adalah keputusan arsitektur yang bagus — logika pemrosesan tidak terduplikasi.

---

## 2. Struktur File Project

```
miniphotoshop/
├── main.pyw                        → Entry point desktop GUI (jalankan tanpa console)
├── run_web.py                      → Entry point web server (uvicorn/FastAPI)
├── requirements.txt                → Dependensi inti (FastAPI, OpenCV, Pillow, numpy)
├── requirements-ml.txt             → Dependensi opsional CNN (TensorFlow)
│
├── mini_photoshop/                 → PAKET UTAMA (logika & GUI desktop)
│   ├── __init__.py
│   ├── image_processor.py          → ★ INTI SEMUA ALGORITMA citra
│   ├── app.py                      → GUI Tkinter (1060 baris)
│   ├── config.py                   → Konstanta UI (warna, ukuran window, dll)
│   ├── history.py                  → Manajemen Undo/Redo
│   ├── ui_components.py            → Widget reusable (slider, scrollable frame)
│   └── ml.py                       → CNN Object Recognition (opsional, TensorFlow)
│
├── backend/
│   └── main.py                     → ★ FastAPI REST API (reuse image_processor)
│
├── frontend/
│   ├── index.html                  → HTML shell aplikasi web
│   ├── app.js                      → Logic JavaScript (state management, API calls)
│   └── styles.css                  → Styling dark/light theme
│
└── tests/
    ├── test_image_processor_smoke.py       → Smoke test semua algoritma
    ├── test_cnn_prediction_format.py       → Test format output CNN
    └── test_large_image_and_cnn_models.py  → Test performa gambar besar
```

**Insight Arsitektur:** Project ini mengikuti prinsip **Separation of Concerns**. `image_processor.py` adalah _pure functions_ — tidak tahu apapun soal GUI atau HTTP. Ini yang memungkinkan backend FastAPI dan desktop Tkinter keduanya memakai algoritma yang sama tanpa copy-paste kode.

---

## 3. Alur Data (Data Flow)

```
[User Upload Gambar]
        ↓
[PIL/Pillow baca file] → [convert ke NumPy array RGB uint8]
        ↓
[image_processor.py] ← fungsi-fungsi pemrosesan dipanggil di sini
        ↓
[NumPy array hasil] → [PIL encode ke PNG/JPEG] → [tampil di UI / kirim ke browser]
```

Semua gambar direpresentasikan sebagai **NumPy array 3D** dengan shape `(height, width, 3)` dan dtype `uint8` (nilai 0–255). Ini adalah representasi standar industri untuk computer vision.

---

## 4. File: `mini_photoshop/image_processor.py` — Inti Semua Algoritma

Inilah file terpenting. Semua operasi matematika ada di sini. Mari kita bedah per kategori.

---

### 4.1 Dataclass `LiveSettings`

```python
@dataclass
class LiveSettings:
    brightness: int = 0
    contrast: int = 0
    rotate: float = 0.0
    scale: float = 1.0
    translate_x: int = 0
    translate_y: int = 0
    hue: int = 0
    saturation: int = 0
    threshold_enabled: bool = False
    threshold_value: int = 127
    interpolation: InterpolationName = "bilinear"
```

Ini adalah **Value Object** — sebuah dataclass yang merangkum semua parameter live editing dari slider UI. Fungsi `apply_live_settings()` menggunakannya untuk menerapkan semua transformasi sekaligus dalam satu pipeline.

---

### 4.2 Kategori Basic — Konversi & Resize

#### RGB ke Grayscale

```python
def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
```

**Rumus asli di balik kode ini:**

OpenCV menerapkan formula weighted luminance (ITU-R BT.601):

```
Y = 0.299 × R + 0.587 × G + 0.114 × B
```

Mengapa bukan rata-rata `(R+G+B)/3`? Karena mata manusia **tidak sama sensitifnya** terhadap semua warna. Mata kita paling sensitif ke hijau (~59%), lalu merah (~30%), baru biru (~11%). Dengan bobot ini, konversi grayscale terasa lebih "alami" secara perseptual.

#### Resize Gambar

```python
def resize_image(image, width, height, interpolation="bilinear"):
    flags = cv2.INTER_NEAREST if interpolation == "nearest" else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=flags)
```

**Dua metode interpolasi yang diimplementasikan:**

| Metode | Rumus | Kapan Dipakai |
|---|---|---|
| **Nearest Neighbor** | `f(x,y) = f(round(x), round(y))` — ambil piksel terdekat | Saat memperbesar pixel art, lebih cepat |
| **Bilinear** | `f(x,y) = (1-a)(1-b)f₀₀ + a(1-b)f₁₀ + (1-a)bf₀₁ + abf₁₁` — interpolasi 4 tetangga terdekat | Default, menghasilkan gambar lebih halus |

---

### 4.3 Kategori Enhancement (Peningkatan Kualitas)

#### Brightness & Contrast — Linear Transform

```python
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    alpha = 1.0 + (contrast / 100.0)
    beta = float(brightness)
    scaled = image.astype(np.float32) * float(alpha) + beta
    return ensure_uint8(scaled)
```

**Rumus asli:**

```
g(x, y) = α · f(x, y) + β
```

Dimana:
- `f(x,y)` = nilai piksel input
- `g(x,y)` = nilai piksel output
- `α` (alpha) = faktor **contrast** (α > 1 → lebih kontras, α < 1 → lebih flat)
- `β` (beta) = faktor **brightness** (β > 0 → lebih terang, β < 0 → lebih gelap)

Ini adalah **Point Operation** paling fundamental dalam pengolahan citra — setiap piksel diproses independen tanpa memperhatikan tetangganya. Perhatikan kode melakukan konversi ke `float32` dulu sebelum operasi — ini penting untuk menghindari integer overflow.

#### Histogram Equalization

```python
def equalize_histogram(image):
    if image.ndim == 2:
        return cv2.equalizeHist(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
```

**Rumus asli (CDF Normalization):**

```
s_k = T(r_k) = (L-1) · Σ p_r(r_j)   untuk j = 0, 1, ..., k
                         j=0
```

Dimana:
- `s_k` = nilai piksel output
- `r_k` = nilai piksel input (0 sampai L-1)
- `p_r(r_j)` = probabilitas kemunculan intensitas r_j = `n_j / N` (jumlah piksel dengan intensitas itu / total piksel)
- `L` = jumlah level abu-abu (biasanya 256)

Pada intinya, ini adalah **transformasi berbasis CDF (Cumulative Distribution Function)** dari histogram. Tujuannya: membuat distribusi histogram menjadi merata/flat sehingga kontras citra meningkat secara global.

**Trick penting di kode:** Untuk citra berwarna, AI tidak langsung equalize channel R, G, B secara terpisah (itu akan merusak warna). Sebagai gantinya, gambar dikonversi ke ruang warna **YCrCb**, lalu hanya channel **Y (luminance/kecerahan)** yang di-equalize. Channel Cr dan Cb (warna) dibiarkan. Ini cara yang benar secara profesional.

#### Sharpening — Unsharp Masking

```python
def unsharp_mask(image, amount=1.0, ksize=5, sigma=1.0):
    ksize = _odd_kernel(ksize)
    blurred = cv2.GaussianBlur(image, (ksize, ksize), float(max(0.0, sigma)))
    return ensure_uint8(cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0))
```

**Rumus asli:**

```
sharpened = original + amount × (original − blurred)
          = (1 + amount) × original − amount × blurred
```

Logika di baliknya: `original - blurred` menghasilkan **"detail mask"** — yaitu informasi tepi/detail yang hilang saat di-blur. Kita tambahkan kembali detail ini ke gambar asli dengan bobot `amount`. Nama "unsharp masking" berasal dari teknik darkroom fotografi analog di mana negatif yang di-blur digunakan sebagai mask.

Bandingkan dengan `sharpen()` yang menggunakan kernel konvolusi statis:
```python
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
```
Ini adalah **Laplacian sharpening kernel** — lebih kasar dan tidak bisa dikontrol parameternya.

#### Smoothing/Blurring

```python
def average_smoothing(image, ksize=5):
    return cv2.blur(image, (ksize, ksize))

def gaussian_blur(image, ksize=5, sigma=0.0):
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)
```

**Average Blur — rumus:**
```
g(x,y) = (1/K²) · Σ f(x+i, y+j)    untuk i,j ∈ [-K/2, K/2]
```
Setiap piksel diganti dengan rata-rata piksel tetangganya dalam window K×K. Sederhana tapi merusak tepi.

**Gaussian Blur — rumus kernel:**
```
G(x,y) = (1 / 2πσ²) · e^(-(x² + y²) / 2σ²)
```
Kernel Gaussian memberi bobot lebih besar ke piksel yang lebih dekat ke pusat. Hasilnya lebih natural karena mengikuti distribusi normal. `sigma` mengontrol "seberapa lebar" blur-nya.

---

### 4.4 Kategori Transform — Transformasi Geometri

#### Affine Transform (Rotate, Scale, Translate)

```python
def affine_transform(image, angle=0.0, scale=1.0, translate_x=0, translate_y=0, interpolation="bilinear"):
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, max(0.01, scale))
    matrix[0, 2] += translate_x
    matrix[1, 2] += translate_y
    flags = cv2.INTER_NEAREST if interpolation == "nearest" else cv2.INTER_LINEAR
    return cv2.warpAffine(image, matrix, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
```

**Rumus Matriks Affine:**

Rotasi + Scaling:
```
M_rot = scale × | cos(θ)   sin(θ)  (1-cos(θ))·cx - sin(θ)·cy |
                 | -sin(θ)  cos(θ)  sin(θ)·cx + (1-cos(θ))·cy |
```

Kemudian `translate_x` dan `translate_y` ditambahkan ke kolom translasi:
```
M[0][2] += translate_x
M[1][2] += translate_y
```

Transformasi dilakukan dengan perkalian matriks:
```
[x']   [M₀₀  M₀₁  M₀₂]   [x]
[y'] = [M₁₀  M₁₁  M₁₂] × [y]
                           [1]
```

`cv2.warpAffine` menerapkan matriks 2×3 ini ke setiap piksel gambar. Rotasi di OpenCV berlawanan dengan konvensi matematika biasa (positif = searah jarum jam) karena sumbu Y di coordinate system image terbalik.

---

### 4.5 Kategori Restoration — Reduksi Noise

#### Median Filter

```python
def median_filter(image, ksize=5):
    ksize = _odd_kernel(ksize)
    return cv2.medianBlur(image, ksize)
```

**Rumus konseptual:**
```
g(x,y) = median{ f(x+i, y+j) }    untuk i,j ∈ [-K/2, K/2]
```

Piksel output adalah **nilai tengah** dari semua piksel di window K×K setelah diurutkan. Ini bukan konvolusi linier — melainkan operasi statistik non-linear. Itulah mengapa median filter sangat efektif untuk **salt-and-pepper noise** (piksel yang tiba-tiba sangat terang atau gelap) — nilai ekstrem tidak mempengaruhi median, berbeda dengan rata-rata yang langsung terpengaruh outlier.

**Perbandingan dengan Gaussian Blur:** Gaussian menghaluskan semua detail termasuk tepi, sementara median filter lebih baik dalam mempertahankan tepi sambil menghilangkan noise impuls.

---

### 4.6 Kategori Binary & Edge — Deteksi Tepi dan Morfologi

#### Threshold Binary

```python
def threshold_binary(image, threshold=127, invert=False):
    gray = to_gray(image)
    kind = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, threshold, 255, kind)
    return gray_to_rgb(binary)
```

**Rumus:**
```
g(x,y) = 255   jika f(x,y) > T
         0     jika f(x,y) ≤ T
```

Operasi paling sederhana dalam binarisasi. Setiap piksel dikategorikan menjadi hitam atau putih berdasarkan threshold T. Dipakai sebagai dasar segmentasi dan preprocessing OCR.

#### Edge Detection — Semua Metode

```python
def edge_detection(image, method="Canny", canny_low=80, canny_high=160, kernel_size=3, log_sigma=0.0):
    gray = to_gray(image)
    # ... dispatch ke masing-masing metode
```

Berikut penjelasan setiap metode yang diimplementasikan:

---

**Sobel**
```python
gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
edges = _normalize_magnitude(gx, gy)
```

Kernel Sobel 3×3:
```
Gx = | -1  0  +1 |    Gy = | +1  +2  +1 |
     | -2  0  +2 |         |  0   0   0 |
     | -1  0  +1 |         | -1  -2  -1 |
```

Magnitude gradient: `M(x,y) = √(Gx² + Gy²)`

Sobel menggunakan pembobotan Gaussian untuk mengurangi sensitivitas terhadap noise dibanding Roberts.

---

**Prewitt**
```python
kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
gx = cv2.filter2D(gray, cv2.CV_64F, kx)
gy = cv2.filter2D(gray, cv2.CV_64F, ky)
```

Kernel Prewitt 3×3:
```
Gx = | -1  0  +1 |    Gy = | +1  +1  +1 |
     | -1  0  +1 |         |  0   0   0 |
     | -1  0  +1 |         | -1  -1  -1 |
```

Mirip Sobel tapi semua bobot bernilai ±1 (tanpa pembobotan Gaussian). Lebih sensitif noise, tapi lebih cepat secara komputasi.

---

**Roberts Cross**
```python
kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
```

Kernel Roberts 2×2:
```
Gx = | +1   0 |    Gy = |  0  +1 |
     |  0  -1 |         | -1   0 |
```

Roberts adalah operator gradient **diagonal** — bekerja pada kernel 2×2. Ini yang paling sederhana dan paling sensitif noise karena tidak ada smoothing sama sekali.

---

**Laplacian**
```python
lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
edges = cv2.convertScaleAbs(lap)
```

Rumus turunan kedua:
```
∇²f = ∂²f/∂x² + ∂²f/∂y²
```

Kernel Laplacian 3×3 (salah satu varian):
```
| 0  -1   0 |
|-1   4  -1 |
| 0  -1   0 |
```

Laplacian mendeteksi tepi dengan mencari **zero-crossing** (pergantian tanda) pada turunan kedua. Sangat sensitif noise karena turunan kedua memperkuat noise. Biasanya perlu di-smooth dulu.

---

**Laplacian of Gaussian (LoG)**
```python
blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), float(max(0.0, log_sigma)))
lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
```

Ini adalah gabungan: **Gaussian Blur dulu, baru Laplacian**. Secara matematis:
```
LoG(x,y) = ∇²G(x,y) * f(x,y)
```

Dimana `G(x,y)` adalah kernel Gaussian. Gaussian mengurangi noise terlebih dahulu sebelum Laplacian diaplikasikan — ini mengatasi kelemahan Laplacian yang sensitif noise.

---

**Canny (paling kompleks)**
```python
blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
edges = cv2.Canny(blurred, low, high)
```

Canny adalah algoritma multi-step:
1. **Gaussian smoothing** — kurangi noise
2. **Gradient computation** — hitung magnitude dan arah gradient (mirip Sobel)
3. **Non-maximum suppression** — tipiskan tepi menjadi 1 piksel saja (hanya local maxima yang dipertahankan)
4. **Double thresholding** — piksel dengan gradient > `high` → strong edge, antara `low`-`high` → weak edge, < `low` → bukan tepi
5. **Edge tracking by hysteresis** — weak edge dipertahankan hanya jika terhubung ke strong edge

Canny dianggap "optimal" karena: deteksi baik (minimal false positive), lokalisasi baik (tepi tepat di posisi yang benar), dan single response (satu tepi = satu respons).

#### Fungsi Helper `_normalize_magnitude`

```python
def _normalize_magnitude(gx, gy):
    mag = np.sqrt(gx * gx + gy * gy)
    if mag.max() <= 0:
        return np.zeros_like(mag, dtype=np.uint8)
    mag = (mag / mag.max()) * 255.0
    return mag.astype(np.uint8)
```

Ini normalisasi magnitude gradient ke range 0–255 untuk bisa ditampilkan sebagai gambar grayscale. Rumus normalisasi: `normalized = (value / max_value) × 255`.

#### Morfologi — Erosion & Dilation

```python
def morphology(image, operation, kernel_size=3, iterations=1):
    gray = to_gray(image)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == "erosion":
        out = cv2.erode(gray, kernel, iterations=max(1, int(iterations)))
    elif operation == "dilation":
        out = cv2.dilate(gray, kernel, iterations=max(1, int(iterations)))
```

**Definisi matematis:**

Erosion (mengikis objek):
```
(f ⊖ b)(x,y) = min{ f(x+i, y+j) }   untuk (i,j) ∈ structuring element b
```
Piksel output = nilai **minimum** dari semua piksel di area structuring element. Efeknya: objek putih mengecil.

Dilation (memperbesar objek):
```
(f ⊕ b)(x,y) = max{ f(x+i, y+j) }   untuk (i,j) ∈ structuring element b
```
Piksel output = nilai **maksimum**. Efeknya: objek putih membesar.

Keduanya merupakan operasi morfologi dasar. Kombinasi erosion-dilation menghasilkan **Opening** (hilangkan noise kecil) dan **Closing** (tutup lubang kecil).

---

### 4.7 Kategori Color Processing

#### Hue/Saturation Adjustment

```python
def adjust_hue_saturation(image, hue_shift=0, saturation_shift=0):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + saturation_shift, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
```

Gambar dikonversi ke ruang warna **HSV (Hue, Saturation, Value)**:
- **H (Hue)** = jenis warna, nilainya 0–179 di OpenCV (bukan 0–360 seperti standar biasanya)
- **S (Saturation)** = kejenuhan warna (0 = abu-abu, 255 = warna penuh)
- **V (Value)** = kecerahan

Mengubah hue di ruang HSV jauh lebih intuitif daripada di RGB. `% 180` memastikan hue "wrap around" (melewati 179 kembali ke 0) — ini sifat melingkar dari warna.

#### Channel Splitting

```python
def split_channel(image, channel):
    idx = {"R": 0, "G": 1, "B": 2}[channel]
    result = np.zeros_like(image)
    result[:, :, idx] = image[:, :, idx]
    return result
```

Buat array kosong (hitam), lalu salin hanya satu channel. Hasilnya adalah gambar yang menampilkan distribusi satu channel warna saja.

---

### 4.8 Kategori Segmentation

#### K-Means Region Segmentation

```python
def region_based_segmentation(image, k=3):
    k = int(np.clip(k, 2, 32))
    data = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(image.shape)
    return segmented
```

**Algoritma K-Means:**
1. `image.reshape((-1, 3))` — ubah gambar dari 3D (H×W×3) menjadi 2D array dengan setiap baris = satu piksel (R,G,B)
2. K-Means mengelompokkan semua piksel menjadi K cluster berdasarkan kedekatan warna dalam ruang warna RGB 3D
3. Setiap piksel diganti dengan warna centroid cluster-nya

**Fungsi objektif K-Means:**
```
minimize Σ Σ ||x_i - μ_k||²
          k  x_i ∈ C_k
```

`cv2.KMEANS_PP_CENTERS` artinya menggunakan inisialisasi **K-Means++** yang lebih cerdas dibanding random — memilih centroid awal yang tersebar jauh satu sama lain untuk konvergensi lebih cepat dan hasil lebih baik.

---

### 4.9 Kategori Compression — Kompresi Citra

#### Simulasi JPEG

```python
def simulate_jpeg(image, quality=50):
    quality = int(np.clip(quality, 1, 100))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
```

Ini mensimulasikan **JPEG lossy compression**. JPEG menggunakan:
1. Konversi RGB → YCbCr
2. **DCT (Discrete Cosine Transform)** per blok 8×8 piksel
3. **Quantization** — koefisien DCT dibagi dengan quantization matrix (langkah yang menyebabkan loss!)
4. Huffman coding untuk kompresi lossless tahap akhir

Semakin rendah quality parameter, semakin agresif quantization-nya, semakin besar loss-nya (artefak "blocky" muncul). Fungsi ini meng-encode ke JPEG lalu decode kembali ke array untuk memperlihatkan efek loss tersebut secara visual.

#### Color Quantization

```python
def quantize_colors(image, levels=4):
    levels = int(np.clip(levels, 2, 32))
    step = 256 // levels
    return ((image // step) * step + step // 2).clip(0, 255).astype(np.uint8)
```

**Rumus:**
```
quantized = floor(pixel / step) × step + step/2
```

Contoh dengan `levels=4`, `step = 256/4 = 64`:
- Piksel 200 → `(200//64)*64 + 32 = 3*64 + 32 = 224`
- Piksel 50 → `(50//64)*64 + 32 = 0*64 + 32 = 32`

Ini mereduksi jumlah warna unik dari 256 level per channel menjadi hanya `levels` nilai. Ini adalah bentuk kompresi lossy sederhana (persis seperti yang dibahas di materi kuantisasi citra).

#### RLE Compression Ratio Estimation

```python
def rle_compression_ratio(image):
    gray = to_gray(image).flatten()
    runs = 1 + np.count_nonzero(gray[1:] != gray[:-1])
    original_size = gray.size
    rle_size = max(1, runs * 2)
    return float(original_size / rle_size)
```

**RLE (Run-Length Encoding)** — kompresi lossless yang menyimpan pasangan (nilai, panjang_run):
```
Original: [10, 10, 10, 10, 200, 200, 50]
RLE:      [(10, 4), (200, 2), (50, 1)]
```

Kode ini menghitung **jumlah run** dengan `np.count_nonzero(gray[1:] != gray[:-1])` — mencari posisi di mana nilai berubah dari piksel sebelumnya. Setiap run membutuhkan 2 byte (nilai + panjang), sehingga `rle_size = runs * 2`.

Rasio > 1 berarti kompresi menguntungkan. Gambar dengan banyak area warna seragam (seperti kartun atau dokumen teks) mendapat rasio tinggi. Foto natural cenderung rasio rendah karena hampir tidak ada run panjang.

---

## 5. File: `mini_photoshop/history.py` — Sistem Undo/Redo

```python
class ImageHistory:
    limit: int = 30
    _undo: list[np.ndarray] = field(default_factory=list)
    _redo: list[np.ndarray] = field(default_factory=list)

    def push(self, image: np.ndarray) -> None:
        self._undo.append(image.copy())
        if len(self._undo) > self.limit:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self, current: np.ndarray) -> Optional[np.ndarray]:
        if not self._undo:
            return None
        self._redo.append(current.copy())
        return self._undo.pop()
```

Ini implementasi **Command Pattern** dengan dua stack. Perhatikan:
- `image.copy()` — wajib! Jika tidak di-copy, semua entry di stack akan merujuk ke array yang sama karena NumPy menggunakan reference semantics
- Saat undo: state saat ini di-push ke redo stack, lalu state dari undo stack di-pop dan dikembalikan
- Saat operasi baru di-apply: redo stack di-clear (tidak bisa redo lagi setelah ada perubahan baru — ini perilaku standar semua editor)
- Ada batas 30 state untuk mencegah memory overflow pada gambar besar

---

## 6. File: `backend/main.py` — FastAPI REST API

### Arsitektur Endpoint

```
GET  /api/health       → health check {"status": "ok"}
GET  /api/features     → daftar semua fitur + kontrolnya (JSON)
POST /api/process      → proses gambar dengan fitur tertentu
POST /api/export       → ekspor gambar ke format tertentu
POST /api/histogram    → hitung histogram gambar
POST /api/cnn          → CNN object recognition
```

### Pola Desain PROCESSORS Dictionary

```python
PROCESSORS: dict[str, Processor] = {
    "brightness_contrast": lambda img, p: (
        ip.adjust_brightness_contrast(img, _as_int(p, "brightness"), _as_int(p, "contrast")),
        "Brightness & Contrast selesai."
    ),
    "jpeg_simulation": lambda img, p: (ip.simulate_jpeg(img, _as_int(p, "quality", 75)), "..."),
    # ... semua fitur lainnya
}
```

Ini adalah **Strategy Pattern** — setiap fitur direpresentasikan sebagai callable yang menerima `(image, params)` dan mengembalikan `(result_image, message)`. Endpoint `/api/process` hanya perlu lookup di dictionary ini:

```python
@app.post("/api/process")
async def process_image(image, feature, params):
    if feature not in PROCESSORS:
        raise HTTPException(404)
    arr = await _read_image(image)
    result, message = PROCESSORS[feature](arr, parsed_params)
    return _image_response(result, message)
```

Keuntungan pola ini: menambahkan fitur baru hanya perlu tambahkan satu entry di `PROCESSORS` dan satu entry di `FEATURES` — tidak perlu ubah routing logic.

### Representasi FEATURES untuk Frontend

```python
FEATURES: list[Feature] = [
    _feature("brightness_contrast", "Brightness & Contrast", "Basic", "...",
        controls=[
            _control("brightness", "Brightness", default=0, minimum=-100, maximum=100),
            _control("contrast", "Contrast", default=0, minimum=-100, maximum=100),
        ],
        presets={"Ringan": {...}, "Sedang": {...}, "Kuat": {...}}
    ),
    ...
]
```

Frontend JavaScript memanggil `GET /api/features` saat startup untuk mendapatkan daftar ini, lalu secara **dinamis** membangun UI (daftar fitur di sidebar, slider parameter, dropdown preset). Ini adalah pendekatan **data-driven UI** — UI dibangun dari data, bukan di-hardcode di HTML.

---

## 7. File: `mini_photoshop/ml.py` — CNN Object Recognition

```python
class CNNRecognizer:
    def __init__(self, model_name="MobileNetV2"):
        spec = MODEL_SPECS[model_name]
        module = import_module(spec.module)
        model_factory = getattr(module, spec.class_name)
        self._model = model_factory(weights="imagenet")

    def predict(self, image_rgb, top_k=5):
        resized = resize(image_rgb, (224, 224)).astype(np.float32)
        batch = np.expand_dims(resized, axis=0)  # shape: (1, 224, 224, 3)
        batch = self._preprocess_input(batch)
        preds = self._model.predict(batch, verbose=0)
        decoded = self._decode_predictions(preds, top=top_k)[0]
        return [Prediction(label=item[1], confidence=float(item[2])) for item in decoded]
```

**Alur CNN prediction:**
1. Resize gambar ke ukuran input model (224×224 untuk MobileNetV2/ResNet50/EfficientNetB0)
2. `np.expand_dims(..., axis=0)` — tambah dimensi batch: shape menjadi `(1, 224, 224, 3)` karena Keras mengharapkan batch
3. `preprocess_input` — normalisasi nilai piksel sesuai kebutuhan masing-masing model (misalnya dari [0-255] ke [-1,1] untuk MobileNetV2)
4. `model.predict()` — forward pass melalui jaringan, output shape `(1, 1000)` (1000 kelas ImageNet)
5. `decode_predictions` — konversi dari indeks kelas ke nama label manusia

Model yang tersedia: **MobileNetV2** (ringan, cocok mobile), **ResNet50** (akurasi tinggi), **EfficientNetB0** (terbaik accuracy-efficiency tradeoff), **InceptionV3** (arsitektur Inception, input 299×299).

Import dilakukan secara **lazy** (hanya saat dipakai) menggunakan `importlib.import_module` — ini agar aplikasi tetap bisa dijalankan tanpa TensorFlow terinstall.

---

## 8. File: `frontend/app.js` — State Management JavaScript

### State Object

```javascript
const state = {
  features: [],
  featureByKey: new Map(),
  selectedFeature: null,
  originalBlob: null,   // gambar asli yang di-upload
  baseBlob: null,       // state terakhir yang di-commit
  previewBlob: null,    // preview sementara (belum di-apply)
  history: [],          // stack undo
  redo: [],             // stack redo
  objectUrls: [],       // URL object untuk garbage collection
  cropStart: null,
  cropEnd: null,
};
```

Frontend menyimpan gambar sebagai **Blob** (binary data) dan menggunakan `URL.createObjectURL()` untuk menampilkannya di `<img>` tag. Ada sistem garbage collection manual (`rememberUrl`) untuk mencegah memory leak dari object URL yang tidak lagi dipakai.

### Live Preview dengan Debouncing

```javascript
let debounceTimer = null;

function scheduleLivePreview() {
  if (!el('livePreview').checked) return;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => processSelected(false), 300);
}
```

Debouncing 300ms mencegah terlalu banyak request ke server saat slider digeser cepat. Ini pattern standar untuk menangani input frekuensi tinggi (scroll, resize, slider, search-as-you-type).

---

## 9. Matriks Fitur vs File vs Rumus

| Materi Kuliah | Fungsi | File | Rumus Kunci |
|---|---|---|---|
| Point Operation | `adjust_brightness_contrast` | `image_processor.py` | `g = α·f + β` |
| Histogram Equalization | `equalize_histogram` | `image_processor.py` | CDF normalization |
| Spatial Filter - Smoothing | `average_smoothing`, `gaussian_blur`, `median_filter` | `image_processor.py` | Konvolusi + kernel |
| Spatial Filter - Sharpening | `sharpen`, `unsharp_mask` | `image_processor.py` | Laplacian kernel, Unsharp formula |
| Transformasi Geometri | `affine_transform`, `resize_image`, `flip_*`, `crop` | `image_processor.py` | Matriks Affine 2×3 |
| Deteksi Tepi | `edge_detection` (6 metode) | `image_processor.py` | ∇f, ∇²f, Canny multi-step |
| Morfologi | `morphology` | `image_processor.py` | Erosion ⊖, Dilation ⊕ |
| Segmentasi | `threshold_segmentation`, `edge_based_segmentation`, `region_based_segmentation` | `image_processor.py` | Thresholding, K-Means |
| Kompresi JPEG | `simulate_jpeg` | `image_processor.py` | DCT + Quantization |
| Kompresi Kuantisasi | `quantize_colors` | `image_processor.py` | `floor(x/step)×step` |
| Kompresi RLE | `rle_compression_ratio` | `image_processor.py` | Run-length counting |
| CNN / Machine Learning | `CNNRecognizer` | `ml.py` | Forward pass CNN (ImageNet) |
| Histogram Analysis | `compute_histograms` | `image_processor.py` | Frekuensi intensitas piksel |

---

## 10. Yang Belum Diimplementasikan (Sesuai SPEK_REP.md)

Berdasarkan dokumen `SPEK_REP.md`, project ini sendiri mengakui bahwa beberapa algoritma kompresi klasik **belum diimplementasikan secara eksplisit**:

- **Huffman Coding** — tidak ada pohon Huffman dari scratch
- **Arithmetic Coding** — tidak ada
- **LZW** — tidak ada

Yang ada hanyalah estimasi RLE dan simulasi JPEG (yang secara internal menggunakan Huffman tapi melalui library, bukan implementasi dari scratch). Jika dosen atau spesifikasi tugas mengharuskan implementasi algoritma-algoritma ini, perlu ditambahkan modul terpisah.

---

## 11. Kesimpulan Arsitektur

Project ini punya struktur yang **solid secara pedagogis**:

1. **Separation of Concerns** — algoritma dipisah dari UI dan transport layer
2. **Single Source of Truth** — `image_processor.py` adalah satu-satunya tempat logika matematika
3. **Strategy Pattern** — `PROCESSORS` dict memungkinkan ekstensi mudah
4. **Data-Driven UI** — frontend dibangun dari deskripsi fitur JSON, bukan hardcode
5. **Lazy Loading** — TensorFlow hanya di-import saat CNN dibutuhkan

Kelemahan yang perlu kamu sadari saat presentasi: tidak ada implementasi Huffman/LZW dari scratch, dan RLE hanya estimasi (bukan encoder/decoder yang bisa mengembalikan data asli).
