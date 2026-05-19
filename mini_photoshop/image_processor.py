"""Core digital image processing helpers for Mini Photoshop Tkinter.

All functions use RGB NumPy arrays with dtype uint8 unless stated otherwise.
The GUI module keeps UI state separate from these pure processing functions so the
operations can be tested without a graphical display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Tuple

import cv2
import numpy as np

InterpolationName = Literal["nearest", "bilinear"]


@dataclass
class LiveSettings:
    """Parameters applied live by the UI sliders."""

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


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Return image clipped to 0..255 and dtype uint8."""

    return np.clip(image, 0, 255).astype(np.uint8)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 3:
        return gray
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def resize_to_fit(image: np.ndarray, max_width: int, max_height: int) -> Tuple[np.ndarray, float, int, int]:
    """Resize image for preview and return (preview, scale, offset_x, offset_y).

    offset values are used by the UI to convert canvas coordinates back to image
    coordinates when the image is centered inside a canvas.
    """

    h, w = image.shape[:2]
    if w == 0 or h == 0:
        return image, 1.0, 0, 0
    scale = min(max_width / w, max_height / h, 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    preview = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return preview, scale, (max_width - new_w) // 2, (max_height - new_h) // 2


def adjust_brightness_contrast(image: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    """Adjust brightness and contrast using alpha-beta linear transform."""

    # Use a linear transform in float space and clamp, avoiding absolute-value
    # behavior from `convertScaleAbs` which can produce unexpected brightening
    # for negative results. Convert to float, apply scale and shift, then
    # clamp to uint8 range.
    alpha = 1.0 + (contrast / 100.0)
    beta = float(brightness)
    scaled = image.astype(np.float32) * float(alpha) + beta
    return ensure_uint8(scaled)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """Histogram equalization. Color images are equalized on luminance channel."""

    if image.ndim == 2:
        return cv2.equalizeHist(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def unsharp_mask(image: np.ndarray, amount: float = 1.0, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Flexible sharpening using unsharp masking."""

    ksize = _odd_kernel(ksize)
    amount = float(np.clip(amount, 0.1, 5.0))
    blurred = cv2.GaussianBlur(image, (ksize, ksize), float(max(0.0, sigma)))
    return ensure_uint8(cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0))


def average_smoothing(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    ksize = _odd_kernel(ksize)
    return cv2.blur(image, (ksize, ksize))


def gaussian_blur(image: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    ksize = _odd_kernel(ksize)
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def median_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    ksize = _odd_kernel(ksize)
    return cv2.medianBlur(image, ksize)


def remove_salt_pepper(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    return median_filter(image, ksize=ksize)


def affine_transform(
    image: np.ndarray,
    angle: float = 0.0,
    scale: float = 1.0,
    translate_x: int = 0,
    translate_y: int = 0,
    interpolation: InterpolationName = "bilinear",
) -> np.ndarray:
    """Rotate, scale, and translate using a 2D affine matrix."""

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, max(0.01, scale))
    matrix[0, 2] += translate_x
    matrix[1, 2] += translate_y
    flags = cv2.INTER_NEAREST if interpolation == "nearest" else cv2.INTER_LINEAR
    return cv2.warpAffine(image, matrix, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 1)


def flip_vertical(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 0)


def crop(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = image.shape[:2]
    left, right = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
    top, bottom = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
    if right - left < 2 or bottom - top < 2:
        raise ValueError("Area crop terlalu kecil.")
    return image[top:bottom, left:right].copy()


def resize_image(image: np.ndarray, width: int, height: int, interpolation: InterpolationName = "bilinear") -> np.ndarray:
    width = max(1, int(width))
    height = max(1, int(height))
    flags = cv2.INTER_NEAREST if interpolation == "nearest" else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=flags)


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    return gray_to_rgb(to_gray(image))


def split_channel(image: np.ndarray, channel: Literal["R", "G", "B"]) -> np.ndarray:
    idx = {"R": 0, "G": 1, "B": 2}[channel]
    result = np.zeros_like(image)
    result[:, :, idx] = image[:, :, idx]
    return result


def adjust_hue_saturation(image: np.ndarray, hue_shift: int = 0, saturation_shift: int = 0) -> np.ndarray:
    if image.ndim == 2:
        return gray_to_rgb(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + saturation_shift, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def threshold_binary(image: np.ndarray, threshold: int = 127, invert: bool = False) -> np.ndarray:
    gray = to_gray(image)
    kind = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, threshold, 255, kind)
    return gray_to_rgb(binary)


def edge_detection(
    image: np.ndarray,
    method: str = "Canny",
    canny_low: int = 80,
    canny_high: int = 160,
    kernel_size: int = 3,
    log_sigma: float = 0.0,
) -> np.ndarray:
    """Edge detection with configurable thresholds and kernel sizes."""

    gray = to_gray(image)
    method_key = method.strip().lower()
    kernel_size = _odd_kernel(kernel_size)
    if kernel_size < 3:
        kernel_size = 3

    if method_key == "canny":
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        low = int(np.clip(canny_low, 0, 255))
        high = int(np.clip(canny_high, 0, 255))
        if high < low:
            low, high = high, low
        edges = cv2.Canny(blurred, low, max(high, low + 1))
    elif method_key == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        edges = _normalize_magnitude(gx, gy)
    elif method_key == "prewitt":
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        gx = cv2.filter2D(gray, cv2.CV_64F, kx)
        gy = cv2.filter2D(gray, cv2.CV_64F, ky)
        edges = _normalize_magnitude(gx, gy)
    elif method_key in {"robert", "roberts"}:
        kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        gx = cv2.filter2D(gray, cv2.CV_64F, kx)
        gy = cv2.filter2D(gray, cv2.CV_64F, ky)
        edges = _normalize_magnitude(gx, gy)
    elif method_key == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
        edges = cv2.convertScaleAbs(lap)
    elif method_key in {"laplacian of gaussian", "log", "laplacian of gaussian (log)"}:
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), float(max(0.0, log_sigma)))
        lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
        edges = cv2.convertScaleAbs(lap)
    else:
        raise ValueError(f"Metode edge tidak dikenal: {method}")

    return gray_to_rgb(edges)


def morphology(image: np.ndarray, operation: Literal["erosion", "dilation"], kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    gray = to_gray(image)
    kernel_size = max(1, int(kernel_size))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == "erosion":
        out = cv2.erode(gray, kernel, iterations=max(1, int(iterations)))
    elif operation == "dilation":
        out = cv2.dilate(gray, kernel, iterations=max(1, int(iterations)))
    else:
        raise ValueError("Operasi morfologi harus erosion atau dilation.")
    return gray_to_rgb(out)


def threshold_segmentation(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    gray = to_gray(image)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    result = image.copy()
    result[mask == 0] = [0, 0, 0]
    return result


def edge_based_segmentation(image: np.ndarray) -> np.ndarray:
    edges = to_gray(edge_detection(image, "Canny"))
    result = image.copy()
    result[edges > 0] = [255, 0, 0]
    return result


def region_based_segmentation(image: np.ndarray, k: int = 3) -> np.ndarray:
    """Simple region segmentation using k-means color clustering."""

    k = int(np.clip(k, 2, 32))
    data = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS) # type: ignore
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(image.shape) # type: ignore
    return segmented


def simulate_jpeg(image: np.ndarray, quality: int = 50) -> np.ndarray:
    quality = int(np.clip(quality, 1, 100))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Gagal melakukan simulasi kompresi JPEG.")
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB) # type: ignore


def quantize_colors(image: np.ndarray, levels: int = 4) -> np.ndarray:
    levels = int(np.clip(levels, 2, 32))
    step = 256 // levels
    return ((image // step) * step + step // 2).clip(0, 255).astype(np.uint8)


def rle_compression_ratio(image: np.ndarray) -> float:
    """Estimate RLE ratio on flattened grayscale values.

    Returns original_size / rle_size. Values > 1 mean compression is beneficial.
    RLE size is estimated as two bytes per run: value + run length token.
    """

    gray = to_gray(image).flatten()
    if gray.size == 0:
        return 1.0
    runs = 1 + np.count_nonzero(gray[1:] != gray[:-1])
    original_size = gray.size
    rle_size = max(1, runs * 2)
    return float(original_size / rle_size)


def compute_histograms(image: np.ndarray) -> dict[str, np.ndarray]:
    """Return grayscale and RGB histograms as arrays of length 256."""

    hist = {"gray": cv2.calcHist([to_gray(image)], [0], None, [256], [0, 256]).flatten()}
    if image.ndim == 3:
        for idx, name in enumerate(("R", "G", "B")):
            hist[name] = cv2.calcHist([image], [idx], None, [256], [0, 256]).flatten()
    return hist


def apply_live_settings(image: np.ndarray, settings: LiveSettings) -> np.ndarray:
    """Apply all live UI parameters in a deterministic pipeline."""

    result = image.copy()
    result = adjust_brightness_contrast(result, settings.brightness, settings.contrast)
    if settings.hue or settings.saturation:
        result = adjust_hue_saturation(result, settings.hue, settings.saturation)
    if settings.rotate or abs(settings.scale - 1.0) > 1e-6 or settings.translate_x or settings.translate_y:
        result = affine_transform(
            result,
            angle=settings.rotate,
            scale=settings.scale,
            translate_x=settings.translate_x,
            translate_y=settings.translate_y,
            interpolation=settings.interpolation,
        )
    if settings.threshold_enabled:
        result = threshold_binary(result, settings.threshold_value)
    return result


def _normalize_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    mag = np.sqrt(gx * gx + gy * gy)
    if mag.max() <= 0:
        return np.zeros_like(mag, dtype=np.uint8)
    mag = (mag / mag.max()) * 255.0
    return mag.astype(np.uint8)


def _odd_kernel(ksize: int) -> int:
    ksize = max(1, int(ksize))
    return ksize if ksize % 2 == 1 else ksize + 1
