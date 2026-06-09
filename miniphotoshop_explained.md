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

#### Flip Horizontal & Flip Vertical

```python
def flip_horizontal(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 1)

def flip_vertical(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 0)
```

`cv2.flip` menerima parameter `flipCode`:
- `flipCode = 1` → flip terhadap **sumbu Y** (kiri-kanan, mirror horizontal)
- `flipCode = 0` → flip terhadap **sumbu X** (atas-bawah, mirror vertikal)
- `flipCode = -1` → flip keduanya sekaligus

**Rumus matematis:**

Flip Horizontal — setiap piksel `(x, y)` dipetakan ke `(W-1-x, y)`:
```
g(x, y) = f(W - 1 - x, y)
```

Flip Vertical — setiap piksel `(x, y)` dipetakan ke `(x, H-1-y)`:
```
g(x, y) = f(x, H - 1 - y)
```

Ini adalah kasus khusus dari transformasi geometri yang bisa diekspresikan sebagai matriks refleksi. Berbeda dengan `affine_transform` yang menghitung matriks floating point dan perlu interpolasi, flip bisa dilakukan dengan **array indexing reversal** langsung — `image[:, ::-1]` untuk horizontal, `image[::-1, :]` untuk vertikal — sehingga lebih efisien. OpenCV melakukan hal yang sama secara internal.

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

#### Salt & Pepper Removal

```python
def remove_salt_pepper(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    return median_filter(image, ksize=ksize)
```

Ini bukan fungsi baru — ini hanya **alias** dari `median_filter` dengan kernel default lebih kecil (3 vs 5). Secara matematis identik. Alasannya ada dua:

1. **Semantik lebih jelas** — pemanggil kode tahu *tujuan* operasinya (remove salt & pepper), bukan hanya *mekanismenya* (median filter). Ini praktik baik dalam desain API.
2. **Default berbeda** — untuk noise salt & pepper, kernel kecil (3×3) biasanya cukup dan lebih aman karena tidak merusak terlalu banyak detail. Kernel 5×5 dipakai kalau noise-nya parah.

**Mengapa median filter adalah solusi terbaik untuk salt-and-pepper noise?**

Salt-and-pepper noise = piksel yang nilainya tiba-tiba melompat ke 0 (pepper, hitam) atau 255 (salt, putih), tidak berkorelasi dengan tetangganya. Jika kamu pakai Gaussian blur, nilai ekstrem 0 atau 255 ini tetap ikut dihitung dalam rata-rata berbobot — sehingga noise tersebar ke tetangga. Dengan median, nilai 0 atau 255 tersebut akan selalu berada di posisi paling pinggir saat diurutkan, dan tidak akan pernah menjadi nilai tengah selama jumlah piksel noise dalam window tidak melebihi 50%.

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

#### Threshold Segmentation

```python
def threshold_segmentation(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    gray = to_gray(image)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    result = image.copy()
    result[mask == 0] = [0, 0, 0]
    return result
```

**Bedakan dengan `threshold_binary`!** Meskipun keduanya pakai threshold, tujuan dan hasilnya berbeda:

| | `threshold_binary` | `threshold_segmentation` |
|---|---|---|
| Kategori | Binary & Edge | Segmentation |
| Output | Gambar hitam-putih murni | Gambar **berwarna** dengan background dihapus |
| Cara kerja | Setiap piksel → 0 atau 255 | Piksel gelap → hitam, piksel terang → **warna aslinya tetap** |

**Rumus / alur:**
```
mask(x,y) = 255   jika gray(x,y) > T     → piksel "foreground"
             0     jika gray(x,y) ≤ T     → piksel "background"

result(x,y) = image(x,y)   jika mask(x,y) == 255
              [0, 0, 0]     jika mask(x,y) == 0
```

`result[mask == 0] = [0, 0, 0]` adalah **boolean indexing NumPy** — semua piksel di mana mask bernilai 0 langsung di-set ke hitam dalam satu operasi vektor tanpa loop. Ini jauh lebih efisien dibanding iterasi per piksel.

Pendekatan ini sangat sederhana dan hanya efektif untuk gambar dengan kontras tinggi antara objek dan background (misalnya dokumen teks di atas kertas putih, atau objek terang di background gelap).

#### Edge-based Segmentation

```python
def edge_based_segmentation(image: np.ndarray) -> np.ndarray:
    edges = to_gray(edge_detection(image, "Canny"))
    result = image.copy()
    result[edges > 0] = [255, 0, 0]
    return result
```

**Rumus / alur:**
1. Jalankan Canny edge detection → hasilkan grayscale edge map (piksel tepi = 255, bukan tepi = 0)
2. Copy gambar asli
3. Di semua posisi yang terdeteksi sebagai tepi, **timpa warna dengan merah [255, 0, 0]**

Hasilnya adalah gambar asli yang **batas-batas objeknya ditandai warna merah**. Ini bukan segmentasi sejati (tidak memisahkan region), melainkan **visualisasi batas region** menggunakan tepi sebagai penanda.

Segmentasi berbasis tepi yang sesungguhnya akan melanjutkan langkah ini dengan operasi seperti *watershed* atau *flood fill* untuk mengisi area di dalam batas tepi — tapi project ini hanya mengimplementasikan langkah penandaan tepinya saja.

`result[edges > 0] = [255, 0, 0]` — lagi-lagi boolean indexing NumPy, kali ini memilih semua piksel di mana nilai edge map > 0.

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

## 9. Cheat Sheet Presentasi — Per Fitur

Format setiap entri: **Materi PCD** → **Kode** (baris per baris) → **Cara Kerja** → **Rumus Asli**

---

### BASIC

---

#### 🔆 Brightness & Contrast

**Materi PCD:** Point Operation / Transformasi Intensitas Linear (Spatial Domain)

**Fungsi:** `adjust_brightness_contrast()` di `image_processor.py`

```python
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    alpha = 1.0 + (contrast / 100.0)   # hitung faktor skala kontras
    beta = float(brightness)            # hitung nilai geser kecerahan
    scaled = image.astype(np.float32) * float(alpha) + beta  # terapkan transformasi linear
    return ensure_uint8(scaled)         # clamp ke 0-255 dan kembalikan ke uint8
```

**Cara kerja baris per baris:**
- `alpha = 1.0 + (contrast / 100.0)` — kontras 0 → alpha=1 (tidak berubah), kontras 50 → alpha=1.5 (lebih kontras), kontras -50 → alpha=0.5 (lebih flat)
- `beta = float(brightness)` — brightness positif = tambah nilai piksel (lebih terang), negatif = kurangi (lebih gelap)
- `image.astype(np.float32)` — wajib konversi ke float32 dulu sebelum perkalian, karena uint8 akan overflow jika nilai melebihi 255
- `* float(alpha) + beta` — terapkan transformasi ke seluruh array sekaligus (vectorized, tidak pakai loop)
- `ensure_uint8()` — clamp nilai ke range 0–255 lalu kembalikan ke dtype uint8

**Rumus asli:**
```
g(x, y) = α · f(x, y) + β
```

---

#### 🖤 RGB → Grayscale

**Materi PCD:** Konversi Ruang Warna / Model Warna

**Fungsi:** `rgb_to_grayscale()` → memanggil `to_gray()` → memanggil `cv2.cvtColor()`

```python
def to_gray(image):
    if image.ndim == 2:          # sudah grayscale, langsung return
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # konversi RGB ke grayscale

def rgb_to_grayscale(image):
    return gray_to_rgb(to_gray(image))  # konversi lalu kembalikan ke 3-channel agar konsisten
```

**Cara kerja baris per baris:**
- `image.ndim == 2` — cek apakah gambar sudah grayscale (2D array) agar tidak diproses ulang
- `cv2.COLOR_RGB2GRAY` — OpenCV menerapkan bobot perceptual ke setiap channel
- `gray_to_rgb()` di akhir — output tetap 3-channel (H×W×3) meski isinya grayscale, supaya konsisten dengan format array di seluruh aplikasi

**Rumus asli:**
```
Y = 0.299·R + 0.587·G + 0.114·B
```

---

#### 📐 Resize / Scaling

**Materi PCD:** Transformasi Geometri + Interpolasi Spasial

**Fungsi:** `resize_image()` di `image_processor.py`

```python
def resize_image(image, width, height, interpolation="bilinear"):
    width = max(1, int(width))    # pastikan dimensi minimal 1 piksel
    height = max(1, int(height))
    flags = cv2.INTER_NEAREST if interpolation == "nearest" else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=flags)
```

**Cara kerja baris per baris:**
- `max(1, int(width))` — guard agar tidak ada dimensi 0 yang akan crash
- `cv2.INTER_NEAREST` — tidak menghitung nilai baru, ambil piksel terdekat saja; cepat tapi hasil "kotak-kotak"
- `cv2.INTER_LINEAR` — bilinear interpolation: hitung rata-rata berbobot dari 4 piksel tetangga; hasil lebih halus
- `cv2.resize(image, (width, height))` — catatan: OpenCV menerima `(width, height)`, bukan `(height, width)` seperti shape NumPy

**Rumus asli (Bilinear Interpolation):**
```
f(x,y) = (1-a)(1-b)·f(x₀,y₀) + a(1-b)·f(x₁,y₀)
        + (1-a)b·f(x₀,y₁)   + ab·f(x₁,y₁)
dimana a = x - x₀,  b = y - y₀
```

---

### ENHANCEMENT

---

#### 📊 Histogram Equalization

**Materi PCD:** Histogram Processing / Peningkatan Kontras Global

**Fungsi:** `equalize_histogram()` di `image_processor.py`

```python
def equalize_histogram(image):
    if image.ndim == 2:                        # jika grayscale, langsung equalize
        return cv2.equalizeHist(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)   # konversi ke ruang warna YCrCb
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0]) # equalize hanya channel Y (luminance)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)    # kembalikan ke RGB
```

**Cara kerja baris per baris:**
- `cv2.COLOR_RGB2YCrCb` — pisahkan informasi kecerahan (Y) dari informasi warna (Cr, Cb); ini standar industri untuk operasi yang hanya ingin ubah kecerahan tanpa merusak warna
- `ycrcb[:, :, 0]` — akses channel pertama (Y = luminance) saja menggunakan NumPy slicing
- `cv2.equalizeHist(ycrcb[:, :, 0])` — hitung histogram → hitung CDF → normalisasi CDF → jadikan fungsi mapping baru untuk setiap piksel
- Channel Cr dan Cb dibiarkan tidak berubah — jika kita equalize R, G, B secara terpisah, warna akan bergeser/rusak

**Rumus asli:**
```
s_k = T(r_k) = (L-1) · Σ p_r(r_j)    j = 0, 1, ..., k
dimana p_r(r_j) = n_j / N
```

---

#### ✨ Sharpening Fleksibel (Unsharp Masking)

**Materi PCD:** Spatial Domain Filtering — High-pass Filter / Image Sharpening

**Fungsi:** `unsharp_mask()` di `image_processor.py`

```python
def unsharp_mask(image, amount=1.0, ksize=5, sigma=1.0):
    ksize = _odd_kernel(ksize)    # pastikan kernel size selalu ganjil (syarat OpenCV)
    amount = float(np.clip(amount, 0.1, 5.0))   # clamp amount ke range valid
    blurred = cv2.GaussianBlur(image, (ksize, ksize), float(max(0.0, sigma)))  # buat versi blur
    return ensure_uint8(cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0))  # kurangi blur dari original
```

**Cara kerja baris per baris:**
- `_odd_kernel(ksize)` — kernel size harus ganjil (3, 5, 7, ...) karena perlu ada piksel pusat; fungsi ini otomatis tambah 1 jika genap
- `cv2.GaussianBlur(...)` — buat salinan gambar yang sudah di-blur; blur ini "menghilangkan" detail/tepi
- `cv2.addWeighted(image, 1.0+amount, blurred, -amount, 0)` — rumusnya: `(1+amount)×original + (-amount)×blurred` = `original + amount×(original - blurred)`
- `original - blurred` = **detail mask** (hanya berisi tepi dan detail)
- Menambahkan detail mask kembali ke original = gambar jadi lebih tajam

**Rumus asli:**
```
sharpened = original + amount × (original − blurred)
          = (1 + amount) × original − amount × blurred
```

---

#### 🌫️ Smoothing / Average Blur

**Materi PCD:** Spatial Domain Filtering — Low-pass Filter / Image Smoothing

**Fungsi:** `average_smoothing()` di `image_processor.py`

```python
def average_smoothing(image, ksize=5):
    ksize = _odd_kernel(ksize)         # pastikan kernel size ganjil
    return cv2.blur(image, (ksize, ksize))  # terapkan average blur dengan kernel K×K
```

**Cara kerja baris per baris:**
- `cv2.blur(image, (ksize, ksize))` — untuk setiap piksel, ambil window K×K di sekitarnya, hitung rata-rata semua piksel dalam window tersebut, jadikan nilai piksel baru
- Efek: noise berkurang karena di-rata-rata dengan tetangga; tapi tepi ikut melunak (ini kelemahan average blur)
- Semakin besar `ksize`, semakin kuat efek smoothing-nya

**Rumus asli:**
```
g(x,y) = (1/K²) · Σ f(x+i, y+j)    untuk i,j ∈ [-K/2, K/2]
```

---

### TRANSFORM

---

#### 🔄 Rotate / Scale / Translate (Affine Transform)

**Materi PCD:** Transformasi Geometri — Affine Transformation

**Fungsi:** `affine_transform()` di `image_processor.py`

```python
def affine_transform(image, angle=0.0, scale=1.0, translate_x=0, translate_y=0, interpolation="bilinear"):
    h, w = image.shape[:2]                              # ambil dimensi gambar
    center = (w / 2.0, h / 2.0)                        # titik pusat rotasi
    matrix = cv2.getRotationMatrix2D(center, angle, max(0.01, scale))  # buat matriks 2×3
    matrix[0, 2] += translate_x                        # tambahkan translasi ke kolom ke-3
    matrix[1, 2] += translate_y
    flags = cv2.INTER_NEAREST if interpolation == "nearest" else cv2.INTER_LINEAR
    return cv2.warpAffine(image, matrix, (w, h), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
```

**Cara kerja baris per baris:**
- `image.shape[:2]` — ambil hanya height dan width, abaikan channel (index 2)
- `cv2.getRotationMatrix2D(center, angle, scale)` — buat matriks rotasi 2×3 yang merotasi gambar di sekitar titik `center` dengan sudut `angle` dan faktor `scale`
- `matrix[0, 2] += translate_x` — kolom ke-3 dari matriks affine adalah vektor translasi; kita tambahkan offset geser secara langsung
- `cv2.warpAffine(image, matrix, (w, h))` — terapkan matriks ke setiap piksel gambar; piksel di luar batas diisi hitam (`borderValue=(0,0,0)`)

**Rumus asli:**
```
[x']   [α   β   (1-α)·cx - β·cy + tx ]   [x]
[y'] = [-β  α   β·cx + (1-α)·cy + ty ] × [y]
                                           [1]
dimana α = scale·cos(θ),  β = scale·sin(θ)
```

---

#### ↔️ Flip Horizontal & Flip Vertical

**Materi PCD:** Transformasi Geometri — Refleksi / Pencerminan

**Fungsi:** `flip_horizontal()` dan `flip_vertical()` di `image_processor.py`

```python
def flip_horizontal(image):
    return cv2.flip(image, 1)   # flipCode=1 → cermin terhadap sumbu Y (kiri-kanan)

def flip_vertical(image):
    return cv2.flip(image, 0)   # flipCode=0 → cermin terhadap sumbu X (atas-bawah)
```

**Cara kerja baris per baris:**
- `cv2.flip(image, 1)` — parameter `1` artinya balik array secara horizontal; setara dengan `image[:, ::-1]` di NumPy (reverse semua kolom)
- `cv2.flip(image, 0)` — parameter `0` artinya balik array secara vertikal; setara dengan `image[::-1, :]` di NumPy (reverse semua baris)
- Tidak ada interpolasi karena tidak ada piksel baru yang dibuat — hanya mengubah urutan piksel yang sudah ada
- Lebih efisien dari affine transform karena tidak perlu menghitung matriks floating point

**Rumus asli:**
```
Flip Horizontal:  g(x, y) = f(W - 1 - x, y)
Flip Vertical:    g(x, y) = f(x, H - 1 - y)
```

---

### RESTORATION

---

#### 🌀 Gaussian Blur

**Materi PCD:** Image Restoration — Linear Spatial Filtering / Noise Reduction

**Fungsi:** `gaussian_blur()` di `image_processor.py`

```python
def gaussian_blur(image, ksize=5, sigma=0.0):
    ksize = _odd_kernel(ksize)    # pastikan kernel size ganjil
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)
```

**Cara kerja baris per baris:**
- `cv2.GaussianBlur(image, (ksize, ksize), sigma)` — hitung kernel Gaussian berukuran `ksize×ksize` dengan standar deviasi `sigma`, lalu konvolusikan kernel tersebut dengan gambar
- `sigma=0.0` — jika sigma=0, OpenCV menghitung sigma otomatis dari ksize: `sigma = 0.3 × ((ksize-1) × 0.5 - 1) + 0.8`
- Gaussian blur lebih unggul dari average blur karena piksel yang lebih dekat ke pusat diberi bobot lebih besar — hasilnya lebih natural
- Digunakan sebagai preprocessing sebelum edge detection (Canny, LoG) untuk mengurangi sensitivitas terhadap noise

**Rumus asli:**
```
G(x,y) = (1 / 2πσ²) · e^(-(x² + y²) / 2σ²)
```

---

#### 📉 Median Filter

**Materi PCD:** Image Restoration — Non-linear Spatial Filtering / Order Statistics Filter

**Fungsi:** `median_filter()` di `image_processor.py`

```python
def median_filter(image, ksize=5):
    ksize = _odd_kernel(ksize)         # pastikan kernel size ganjil
    return cv2.medianBlur(image, ksize)  # terapkan median filter
```

**Cara kerja baris per baris:**
- `cv2.medianBlur(image, ksize)` — untuk setiap piksel, ambil semua nilai piksel di window K×K, urutkan dari kecil ke besar, ambil nilai tengah (median), jadikan nilai piksel baru
- Ini **bukan konvolusi** — tidak ada perkalian kernel, hanya operasi sort dan ambil nilai tengah
- Nilai ekstrem (0 atau 255 dari salt-and-pepper noise) akan selalu berada di posisi paling pinggir setelah sorting — tidak akan pernah jadi median, sehingga noise impuls hilang sempurna
- Tepi objek lebih terjaga dibanding Gaussian blur karena tidak ada rata-rata yang "mencampurkan" piksel tepi

**Rumus asli:**
```
g(x,y) = median{ f(x+i, y+j) }    untuk i,j ∈ [-K/2, K/2]
```

---

#### 🧂 Salt & Pepper Removal

**Materi PCD:** Image Restoration — Impulse Noise Removal

**Fungsi:** `remove_salt_pepper()` di `image_processor.py`

```python
def remove_salt_pepper(image, ksize=3):
    return median_filter(image, ksize=ksize)   # alias median_filter dengan kernel default lebih kecil
```

**Cara kerja baris per baris:**
- Fungsi ini adalah **alias semantik** dari `median_filter()` — secara matematis identik
- `ksize=3` default lebih kecil dari `median_filter()` yang default-nya 5; untuk noise salt & pepper, kernel 3×3 sudah cukup efektif dan lebih aman karena tidak terlalu menghapus detail
- Alasan dipisah jadi fungsi sendiri: penamaan yang jelas membantu pembaca kode memahami *tujuan* operasi, bukan hanya mekanismenya — ini prinsip clean code

**Rumus asli:** sama dengan Median Filter:
```
g(x,y) = median{ f(x+i, y+j) }    untuk i,j ∈ [-K/2, K/2]
```

---

### BINARY & EDGE

---

#### ⬛ Threshold Binary

**Materi PCD:** Binarisasi Citra / Global Thresholding

**Fungsi:** `threshold_binary()` di `image_processor.py`

```python
def threshold_binary(image, threshold=127, invert=False):
    gray = to_gray(image)    # konversi ke grayscale dulu
    kind = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY  # pilih mode normal atau invert
    _, binary = cv2.threshold(gray, threshold, 255, kind)  # terapkan threshold
    return gray_to_rgb(binary)   # kembalikan ke format 3-channel
```

**Cara kerja baris per baris:**
- `to_gray(image)` — threshold hanya bisa dilakukan pada gambar grayscale (satu channel intensitas)
- `cv2.THRESH_BINARY` — piksel > threshold → 255 (putih), piksel ≤ threshold → 0 (hitam)
- `cv2.THRESH_BINARY_INV` — kebalikannya: piksel > threshold → 0, piksel ≤ threshold → 255
- `cv2.threshold(gray, threshold, 255, kind)` — mengembalikan dua nilai: `_` (nilai threshold optimal, diabaikan di sini) dan `binary` (gambar hasil)
- `gray_to_rgb(binary)` — konversi kembali ke 3-channel agar format konsisten dengan seluruh pipeline

**Rumus asli:**
```
g(x,y) = 255   jika f(x,y) > T
          0    jika f(x,y) ≤ T
```

---

#### 🔍 Edge Detection — Sobel

**Materi PCD:** Deteksi Tepi — First-order Gradient Operator

**Fungsi:** `edge_detection(method="Sobel")` di `image_processor.py`

```python
gray = to_gray(image)
gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)  # gradient arah X
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)  # gradient arah Y
edges = _normalize_magnitude(gx, gy)                        # hitung magnitude
```

**Cara kerja baris per baris:**
- `cv2.Sobel(gray, cv2.CV_64F, 1, 0)` — parameter `1,0` artinya turunan ke-1 di arah X, turunan ke-0 di arah Y (horizontal edge)
- `cv2.Sobel(gray, cv2.CV_64F, 0, 1)` — parameter `0,1` artinya turunan ke-0 di X, turunan ke-1 di Y (vertical edge)
- `cv2.CV_64F` — hasil disimpan dalam float64 karena nilai gradient bisa negatif (piksel gelap ke terang vs terang ke gelap)
- `_normalize_magnitude(gx, gy)` — hitung `√(gx²+gy²)` lalu normalisasi ke range 0–255

**Rumus asli:**
```
Gx = [-1  0  +1]     Gy = [+1  +2  +1]
     [-2  0  +2]          [ 0   0   0]
     [-1  0  +1]          [-1  -2  -1]

M(x,y) = √(Gx² + Gy²)
```

---

#### 🔍 Edge Detection — Prewitt

**Materi PCD:** Deteksi Tepi — First-order Gradient Operator

**Fungsi:** `edge_detection(method="Prewitt")` di `image_processor.py`

```python
gray = to_gray(image)
kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)   # kernel Prewitt horizontal
ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)   # kernel Prewitt vertikal
gx = cv2.filter2D(gray, cv2.CV_64F, kx)    # konvolusi manual dengan kernel kx
gy = cv2.filter2D(gray, cv2.CV_64F, ky)    # konvolusi manual dengan kernel ky
edges = _normalize_magnitude(gx, gy)
```

**Cara kerja baris per baris:**
- Berbeda dari Sobel, Prewitt tidak punya fungsi khusus di OpenCV — kernelnya didefinisikan manual sebagai array NumPy
- `cv2.filter2D(gray, cv2.CV_64F, kx)` — fungsi konvolusi umum; terapkan kernel `kx` ke gambar `gray`, simpan dalam float64
- Kernel Prewitt: semua bobot bernilai ±1 (tidak ada pembobotan Gaussian seperti Sobel)
- Lebih sederhana dari Sobel tapi lebih sensitif terhadap noise karena tidak ada smoothing bawaan

**Rumus asli:**
```
Gx = [-1  0  +1]     Gy = [+1  +1  +1]
     [-1  0  +1]          [ 0   0   0]
     [-1  0  +1]          [-1  -1  -1]

M(x,y) = √(Gx² + Gy²)
```

---

#### 🔍 Edge Detection — Roberts Cross

**Materi PCD:** Deteksi Tepi — First-order Gradient Operator (Diagonal)

**Fungsi:** `edge_detection(method="Roberts")` di `image_processor.py`

```python
gray = to_gray(image)
kx = np.array([[1, 0], [0, -1]], dtype=np.float32)    # kernel Roberts diagonal 1
ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)    # kernel Roberts diagonal 2
gx = cv2.filter2D(gray, cv2.CV_64F, kx)
gy = cv2.filter2D(gray, cv2.CV_64F, ky)
edges = _normalize_magnitude(gx, gy)
```

**Cara kerja baris per baris:**
- Roberts menggunakan kernel **2×2** — paling kecil di antara semua operator deteksi tepi
- `kx = [[1,0],[0,-1]]` — mendeteksi perbedaan diagonal dari kiri-atas ke kanan-bawah
- `ky = [[0,1],[-1,0]]` — mendeteksi perbedaan diagonal dari kanan-atas ke kiri-bawah
- Karena kernel kecil dan tidak ada smoothing, Roberts sangat sensitif terhadap noise — jarang dipakai di industri
- Cocok untuk gambar yang sudah bersih/smooth

**Rumus asli:**
```
Gx = [+1   0]     Gy = [ 0  +1]
     [ 0  -1]          [-1   0]

M(x,y) = √(Gx² + Gy²)
```

---

#### 🔍 Edge Detection — Laplacian

**Materi PCD:** Deteksi Tepi — Second-order Derivative Operator

**Fungsi:** `edge_detection(method="Laplacian")` di `image_processor.py`

```python
gray = to_gray(image)
kernel_size = _odd_kernel(kernel_size)
lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)  # hitung turunan kedua
edges = cv2.convertScaleAbs(lap)   # ambil nilai absolut dan konversi ke uint8
```

**Cara kerja baris per baris:**
- `cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)` — hitung turunan kedua (`∂²f/∂x² + ∂²f/∂y²`) menggunakan kernel Laplacian
- Laplacian menggunakan **turunan kedua** berbeda dengan Sobel/Prewitt/Roberts yang pakai turunan pertama
- Hasilnya bisa negatif (zero-crossing), maka disimpan dalam float64
- `cv2.convertScaleAbs(lap)` — ambil nilai absolut (|negatif| → positif) lalu convert ke uint8 — ini berbeda dari `_normalize_magnitude` karena tidak me-rescale ke 0–255, hanya clip absolut
- Sangat sensitif noise — jarang dipakai tanpa smoothing sebelumnya

**Rumus asli:**
```
∇²f = ∂²f/∂x² + ∂²f/∂y²

Kernel 3×3:
[ 0  -1   0]
[-1   4  -1]
[ 0  -1   0]
```

---

#### 🔍 Edge Detection — LoG (Laplacian of Gaussian)

**Materi PCD:** Deteksi Tepi — Laplacian of Gaussian / Marr-Hildreth Operator

**Fungsi:** `edge_detection(method="Laplacian of Gaussian")` di `image_processor.py`

```python
gray = to_gray(image)
kernel_size = _odd_kernel(kernel_size)
blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), float(max(0.0, log_sigma)))  # step 1: smooth dulu
lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)  # step 2: baru laplacian
edges = cv2.convertScaleAbs(lap)
```

**Cara kerja baris per baris:**
- LoG = **dua langkah**: Gaussian blur dulu, kemudian Laplacian
- `cv2.GaussianBlur(...)` — sigma mengontrol seberapa kuat noise direduksi sebelum deteksi tepi; sigma besar = lebih banyak smoothing
- `cv2.Laplacian(blurred, ...)` — setelah noise berkurang, baru Laplacian diterapkan ke gambar yang sudah smooth
- Ini mengatasi kelemahan Laplacian biasa yang sensitif noise — Gaussian "melindungi" Laplacian dari noise
- Secara matematis, LoG = konvolusi kernel LoG sekaligus, tapi implementasi di sini melakukannya dua tahap yang hasilnya sama

**Rumus asli:**
```
LoG(x,y) = ∇²[G(x,y) * f(x,y)]
          = [∇²G(x,y)] * f(x,y)

∇²G(x,y) = (1/πσ⁴) · [1 - (x²+y²)/2σ²] · e^(-(x²+y²)/2σ²)
```

---

#### 🔍 Edge Detection — Canny

**Materi PCD:** Deteksi Tepi — Optimal Edge Detector (Multi-step)

**Fungsi:** `edge_detection(method="Canny")` di `image_processor.py`

```python
gray = to_gray(image)
blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)   # step 1: Gaussian smoothing
low = int(np.clip(canny_low, 0, 255))
high = int(np.clip(canny_high, 0, 255))
if high < low:
    low, high = high, low    # pastikan low < high
edges = cv2.Canny(blurred, low, max(high, low + 1))  # step 2-5: Canny pipeline
```

**Cara kerja baris per baris:**
- `cv2.GaussianBlur(gray, (5,5), 1.4)` — step 1: smooth untuk kurangi noise; sigma=1.4 adalah nilai umum untuk Canny
- `if high < low: low, high = high, low` — swap otomatis jika user keliru memasukkan nilai; mencegah error
- `cv2.Canny(blurred, low, high)` — menjalankan 4 step sekaligus di dalam OpenCV:
  - **Step 2:** Hitung gradient magnitude dan arah menggunakan Sobel
  - **Step 3:** Non-maximum suppression — tipiskan tepi menjadi 1 piksel
  - **Step 4:** Double thresholding — `> high` = strong edge, antara `low-high` = weak edge, `< low` = bukan tepi
  - **Step 5:** Edge tracking by hysteresis — weak edge dipertahankan hanya jika terhubung ke strong edge

**Rumus asli:** Canny tidak punya satu rumus tunggal — ini adalah algoritma multi-step. Kriteria optimalitas Canny:
```
1. Good detection:    minimal false positive dan false negative
2. Good localization: tepi terdeteksi tepat di posisi yang benar
3. Single response:   satu tepi menghasilkan satu respons saja
```

---

#### 🔲 Morphology — Erosion & Dilation

**Materi PCD:** Morphological Image Processing

**Fungsi:** `morphology()` di `image_processor.py`

```python
def morphology(image, operation, kernel_size=3, iterations=1):
    gray = to_gray(image)                                    # konversi ke grayscale
    kernel_size = max(1, int(kernel_size))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # buat structuring element kotak
    if operation == "erosion":
        out = cv2.erode(gray, kernel, iterations=max(1, int(iterations)))    # erosi
    elif operation == "dilation":
        out = cv2.dilate(gray, kernel, iterations=max(1, int(iterations)))   # dilasi
    return gray_to_rgb(out)
```

**Cara kerja baris per baris:**
- `np.ones((kernel_size, kernel_size), np.uint8)` — buat structuring element berbentuk kotak penuh berisi 1; bentuk SE menentukan "arah" operasi morfologi
- `cv2.erode(gray, kernel, iterations=...)` — untuk setiap piksel, ambil nilai **minimum** dari semua piksel di bawah structuring element; efek: objek putih mengecil, noise kecil hilang
- `cv2.dilate(gray, kernel, iterations=...)` — untuk setiap piksel, ambil nilai **maksimum**; efek: objek putih membesar, celah kecil tertutup
- `iterations` — jumlah kali operasi diulang; erosion 2 kali = erosi lebih dalam dari 1 kali

**Rumus asli:**
```
Erosion:  (f ⊖ b)(x,y) = min{ f(x+i, y+j) | (i,j) ∈ b }
Dilation: (f ⊕ b)(x,y) = max{ f(x+i, y+j) | (i,j) ∈ b }
```

---

### COLOR

---

#### 🎨 Channel Splitting RGB

**Materi PCD:** Model Warna RGB — Dekomposisi Channel

**Fungsi:** `split_channel()` di `image_processor.py`

```python
def split_channel(image, channel):
    idx = {"R": 0, "G": 1, "B": 2}[channel]  # petakan nama channel ke index array
    result = np.zeros_like(image)              # buat array kosong (hitam) dengan shape sama
    result[:, :, idx] = image[:, :, idx]       # salin hanya channel yang dipilih
    return result
```

**Cara kerja baris per baris:**
- `{"R": 0, "G": 1, "B": 2}[channel]` — dictionary lookup: channel "R" → index 0, "G" → 1, "B" → 2 (urutan sesuai format RGB yang dipakai project ini)
- `np.zeros_like(image)` — buat array berisi nol dengan shape dan dtype yang sama dengan input; hasilnya gambar hitam total
- `result[:, :, idx] = image[:, :, idx]` — NumPy slicing: salin semua baris, semua kolom, hanya channel ke-`idx` dari image ke result
- Channel yang tidak dipilih tetap 0 (hitam) — hasilnya gambar yang hanya menampilkan distribusi satu warna

**Rumus asli:**
```
result_R(x,y) = [f_R(x,y), 0, 0]
result_G(x,y) = [0, f_G(x,y), 0]
result_B(x,y) = [0, 0, f_B(x,y)]
```

---

#### 🌈 Hue / Saturation

**Materi PCD:** Konversi Ruang Warna — Model Warna HSV

**Fungsi:** `adjust_hue_saturation()` di `image_processor.py`

```python
def adjust_hue_saturation(image, hue_shift=0, saturation_shift=0):
    if image.ndim == 2:
        return gray_to_rgb(image)   # grayscale tidak punya hue/saturation, langsung return
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)  # konversi ke HSV, pakai int16
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180               # geser hue dengan wrap-around
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + saturation_shift, 0, 255)  # geser saturasi dengan clamp
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)    # kembalikan ke RGB
```

**Cara kerja baris per baris:**
- `cv2.COLOR_RGB2HSV` — konversi ke ruang warna HSV: H (jenis warna), S (kejenuhan), V (kecerahan)
- `.astype(np.int16)` — wajib int16 (bukan uint8) karena operasi penjumlahan bisa menghasilkan nilai negatif sementara sebelum di-clamp/modulo
- `(hsv[:, :, 0] + hue_shift) % 180` — hue di OpenCV nilainya 0–179 (bukan 0–360); `% 180` memastikan nilai wrap-around: misal 170 + 20 = 190 → 190%180 = 10 (kembali ke awal spektrum warna)
- `np.clip(... , 0, 255)` — saturasi tidak bisa wrap-around seperti hue; harus di-clamp ke 0–255
- `hsv.astype(np.uint8)` — kembalikan ke uint8 sebelum konversi warna balik ke RGB

**Rumus asli:**
```
H ∈ [0°, 360°]  → di OpenCV H ∈ [0, 179]
S ∈ [0, 1]      → di OpenCV S ∈ [0, 255]
V ∈ [0, 1]      → di OpenCV V ∈ [0, 255]

H_new = (H + shift) mod 180
S_new = clamp(S + shift, 0, 255)
```

---

### SEGMENTATION

---

#### ✂️ Threshold Segmentation

**Materi PCD:** Segmentasi Citra — Threshold-based Segmentation

**Fungsi:** `threshold_segmentation()` di `image_processor.py`

```python
def threshold_segmentation(image, threshold=127):
    gray = to_gray(image)                                              # konversi ke grayscale untuk thresholding
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)  # buat binary mask
    result = image.copy()                                              # salin gambar asli (berwarna)
    result[mask == 0] = [0, 0, 0]                                     # piksel background → hitam
    return result
```

**Cara kerja baris per baris:**
- `to_gray(image)` — threshold dilakukan pada intensitas grayscale, bukan per channel warna
- `cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)` — hasilkan mask: piksel > threshold → 255 (foreground), piksel ≤ threshold → 0 (background)
- `result = image.copy()` — salin gambar **berwarna** asli, bukan grayscale; di sinilah bedanya dengan `threshold_binary`
- `result[mask == 0] = [0, 0, 0]` — boolean indexing NumPy: semua piksel di mana mask bernilai 0 langsung di-set hitam dalam satu operasi; foreground tetap warna aslinya

**Rumus asli:**
```
mask(x,y)   = 255  jika gray(x,y) > T
               0   jika gray(x,y) ≤ T

result(x,y) = image(x,y)   jika mask(x,y) = 255  (foreground tetap berwarna)
              [0, 0, 0]     jika mask(x,y) = 0    (background jadi hitam)
```

---

#### ✂️ Edge-based Segmentation

**Materi PCD:** Segmentasi Citra — Edge-based Segmentation

**Fungsi:** `edge_based_segmentation()` di `image_processor.py`

```python
def edge_based_segmentation(image):
    edges = to_gray(edge_detection(image, "Canny"))  # deteksi tepi dengan Canny, ambil grayscale-nya
    result = image.copy()                            # salin gambar asli
    result[edges > 0] = [255, 0, 0]                 # piksel tepi → merah
    return result
```

**Cara kerja baris per baris:**
- `edge_detection(image, "Canny")` — jalankan Canny edge detection; hasilnya gambar di mana piksel tepi = 255, bukan tepi = 0
- `to_gray(...)` — konversi output Canny ke grayscale (sebenarnya sudah grayscale, ini untuk memastikan shape 2D)
- `result = image.copy()` — salin gambar **berwarna** asli
- `result[edges > 0] = [255, 0, 0]` — boolean indexing: semua piksel yang terdeteksi sebagai tepi ditimpa warna merah

**Catatan penting untuk presentasi:** Ini adalah **visualisasi batas region**, bukan segmentasi penuh. Segmentasi edge-based yang lengkap akan melanjutkan dengan flood fill atau watershed untuk mengisi area di dalam batas — tapi project ini hanya menampilkan batasnya saja.

**Rumus asli:**
```
edges(x,y) = Canny(f(x,y))

result(x,y) = [255, 0, 0]   jika edges(x,y) > 0   (batas region → merah)
              image(x,y)     jika edges(x,y) = 0   (interior region → warna asli)
```

---

#### ✂️ Region-based Segmentation (K-Means)

**Materi PCD:** Segmentasi Citra — Region-based Segmentation / Clustering

**Fungsi:** `region_based_segmentation()` di `image_processor.py`

```python
def region_based_segmentation(image, k=3):
    k = int(np.clip(k, 2, 32))                                     # clamp k ke range valid
    data = image.reshape((-1, 3)).astype(np.float32)               # ubah gambar jadi daftar piksel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)  # kondisi berhenti
    _compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)                                     # konversi centroid ke uint8
    segmented = centers[labels.flatten()].reshape(image.shape)     # ganti setiap piksel dengan warna centroid
    return segmented
```

**Cara kerja baris per baris:**
- `image.reshape((-1, 3))` — ubah array 3D (H×W×3) menjadi 2D (H×W, 3); setiap baris = satu piksel dengan nilai [R,G,B]; `-1` artinya hitung otomatis
- `cv2.kmeans(data, k, ...)` — jalankan K-Means: kelompokkan semua piksel menjadi K cluster berdasarkan kedekatan warna di ruang RGB 3D
- `criteria = (... , 20, 1.0)` — hentikan iterasi jika: sudah 20 iterasi ATAU perubahan centroid < 1.0 (epsilon)
- `cv2.KMEANS_PP_CENTERS` — inisialisasi K-Means++ (lebih cerdas dari random)
- `centers[labels.flatten()]` — fancy indexing: ganti setiap piksel dengan warna centroid cluster-nya
- `.reshape(image.shape)` — kembalikan ke shape gambar asli (H×W×3)

**Rumus asli:**
```
minimize  Σ   Σ    ||x_i − μ_k||²
          k  x_i∈Cₖ

dimana μ_k = (1/|Cₖ|) · Σ x_i   (centroid cluster k)
                          x_i∈Cₖ
```

---

### COMPRESSION

---

#### 📦 Simulasi JPEG Quality

**Materi PCD:** Kompresi Citra Lossy — JPEG Compression

**Fungsi:** `simulate_jpeg()` di `image_processor.py`

```python
def simulate_jpeg(image, quality=50):
    quality = int(np.clip(quality, 1, 100))                          # clamp quality ke 1-100
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)                     # konversi ke BGR (format OpenCV)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])  # encode ke JPEG
    if not ok:
        raise ValueError("Gagal melakukan simulasi kompresi JPEG.")
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)            # decode balik dari JPEG
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)              # kembalikan ke RGB
```

**Cara kerja baris per baris:**
- `cv2.COLOR_RGB2BGR` — OpenCV menyimpan gambar dalam format BGR (bukan RGB); perlu dikonversi sebelum encoding
- `cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])` — encode array ke format JPEG dalam memori (bukan file); `quality` menentukan agresivitas quantization
- `ok` — boolean apakah encoding berhasil; dicek sebelum lanjut
- `cv2.imdecode(encoded, cv2.IMREAD_COLOR)` — decode kembali dari JPEG bytes ke array; di sinilah **loss terlihat** — data yang hilang saat quantization tidak bisa dikembalikan
- Encode → decode ini memperlihatkan artefak JPEG (blocky, banding) secara visual

Pipeline JPEG di balik kode ini:
```
RGB → YCbCr → DCT per blok 8×8 → Quantization (LOSSY) → Huffman → [bytes]
[bytes] → Huffman decode → Dequantize → IDCT → YCbCr → RGB
```

**Rumus asli (DCT 2D):**
```
F(u,v) = (1/4)·C(u)·C(v) · Σ Σ f(x,y)·cos[(2x+1)uπ/16]·cos[(2y+1)vπ/16]
                             x y
dimana C(0) = 1/√2, C(u) = 1 untuk u > 0
```

---

#### 🎨 Color Quantization

**Materi PCD:** Kompresi Citra — Kuantisasi Warna / Color Depth Reduction

**Fungsi:** `quantize_colors()` di `image_processor.py`

```python
def quantize_colors(image, levels=4):
    levels = int(np.clip(levels, 2, 32))   # clamp levels ke range valid
    step = 256 // levels                   # hitung lebar setiap interval kuantisasi
    return ((image // step) * step + step // 2).clip(0, 255).astype(np.uint8)
```

**Cara kerja baris per baris:**
- `step = 256 // levels` — bagi range 0–255 menjadi `levels` interval sama lebar; contoh: levels=4 → step=64
- `image // step` — integer division: petakan setiap piksel ke indeks interval-nya; contoh: piksel 200 → 200//64 = 3
- `* step` — kembalikan ke nilai asli (awal interval); contoh: 3 × 64 = 192
- `+ step // 2` — geser ke tengah interval (representasi terbaik untuk interval itu); contoh: 192 + 32 = 224
- `.clip(0, 255).astype(np.uint8)` — pastikan tidak ada overflow
- Seluruh operasi dilakukan **vectorized** pada array NumPy — berlaku ke semua piksel dan semua channel sekaligus

**Rumus asli:**
```
step = ⌊256 / levels⌋
quantized(x,y) = ⌊f(x,y) / step⌋ × step + ⌊step / 2⌋
```

---

#### 📏 RLE Compression Ratio

**Materi PCD:** Kompresi Citra Lossless — Run-Length Encoding

**Fungsi:** `rle_compression_ratio()` di `image_processor.py`

```python
def rle_compression_ratio(image):
    gray = to_gray(image).flatten()    # konversi ke grayscale lalu jadikan array 1D
    if gray.size == 0:
        return 1.0
    runs = 1 + np.count_nonzero(gray[1:] != gray[:-1])  # hitung jumlah run
    original_size = gray.size                            # ukuran data asli (dalam piksel)
    rle_size = max(1, runs * 2)                          # estimasi ukuran RLE (2 byte per run)
    return float(original_size / rle_size)               # hitung rasio kompresi
```

**Cara kerja baris per baris:**
- `to_gray(image).flatten()` — RLE dihitung pada data 1D; `.flatten()` ubah array 2D (H×W) menjadi array 1D panjang H×W
- `gray[1:] != gray[:-1]` — bandingkan setiap elemen dengan elemen sebelumnya; hasilkan array boolean: True di mana nilai berubah
- `np.count_nonzero(...)` — hitung jumlah perubahan nilai = jumlah batas antar run
- `runs = 1 + jumlah_perubahan` — +1 karena run pertama tidak punya perubahan sebelumnya
- `rle_size = runs * 2` — estimasi: setiap run butuh 2 byte (1 byte nilai + 1 byte panjang run)
- `original_size / rle_size` — rasio > 1 berarti RLE menguntungkan (data lebih kecil setelah kompresi)

**Rumus asli:**
```
RLE encoding: (value, run_length) per run
Contoh: [10,10,10,200,200,50] → [(10,3),(200,2),(50,1)]

Compression Ratio = Original Size / Compressed Size
                  = N / (runs × 2)
```

---

### MACHINE LEARNING

---

#### 🤖 CNN Object Recognition

**Materi PCD:** Machine Learning untuk Pengolahan Citra — Convolutional Neural Network

**Fungsi:** `CNNRecognizer.predict()` di `mini_photoshop/ml.py`

```python
def predict(self, image_rgb, top_k=5):
    top_k = max(1, min(int(top_k), 10))
    width, height = self.spec.input_size                              # ukuran input model (224×224 atau 299×299)
    resized = resize(image_rgb, (width, height)).astype(np.float32)  # resize gambar ke ukuran input CNN
    batch = np.expand_dims(resized, axis=0)                          # tambah dimensi batch: (1, H, W, 3)
    batch = self._preprocess_input(batch)                            # normalisasi nilai piksel
    preds = self._model.predict(batch, verbose=0)                    # forward pass CNN
    decoded = self._decode_predictions(preds, top=top_k)[0]         # ambil top-K prediksi
    return [Prediction(label=item[1], confidence=float(item[2])) for item in decoded]
```

**Cara kerja baris per baris:**
- `resize(image_rgb, (width, height))` — CNN punya fixed input size; MobileNetV2/ResNet50/EfficientNetB0 = 224×224, InceptionV3 = 299×299
- `.astype(np.float32)` — model Keras mengharapkan float32, bukan uint8
- `np.expand_dims(resized, axis=0)` — tambah dimensi batch di axis 0: shape dari (224,224,3) menjadi (1,224,224,3); Keras selalu mengharapkan input dalam batch
- `self._preprocess_input(batch)` — normalisasi berbeda per model; MobileNetV2: `[-1, 1]`, ResNet50: subtract mean ImageNet
- `self._model.predict(batch)` — jalankan forward pass melalui seluruh layer CNN; output shape (1, 1000) = probabilitas 1000 kelas ImageNet
- `self._decode_predictions(preds, top=top_k)[0]` — konversi indeks kelas ke nama label manusia, ambil K probabilitas tertinggi

**Model yang tersedia dan karakteristiknya:**
```
MobileNetV2   → ringan, cocok untuk device terbatas, akurasi cukup baik
ResNet50      → akurasi tinggi, arsitektur residual connection
EfficientNetB0→ terbaik accuracy-efficiency tradeoff
InceptionV3   → input 299×299, arsitektur inception module
```

**Rumus asli (Softmax output layer CNN):**
```
P(class_k | x) = e^(z_k) / Σ e^(z_j)    untuk semua j ∈ 1000 kelas
```

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
