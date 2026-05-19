"""FastAPI backend for the Mini Photoshop web version.

The backend intentionally reuses the original ``mini_photoshop.image_processor``
module so the image-processing algorithms stay in one place and remain testable
without a GUI.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Callable, cast, Literal
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFile

from mini_photoshop import image_processor as ip
from mini_photoshop.ml import DEFAULT_MODEL_NAME, MODEL_NAMES

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Mini Photoshop Web API",
    description="Backend FastAPI untuk pengolahan citra Mini Photoshop.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Image-Message", "X-Image-Width", "X-Image-Height"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


Control = dict[str, Any]
Feature = dict[str, Any]
Processor = Callable[[np.ndarray, dict[str, Any]], tuple[np.ndarray, str]]


def _control(
    key: str,
    label: str,
    *,
    kind: str = "slider",
    default: Any = 0,
    minimum: float = 0,
    maximum: float = 100,
    options: list[str] | None = None,
    integer: bool = True,
    help_text: str = "",
) -> Control:
    return {
        "key": key,
        "label": label,
        "kind": kind,
        "default": default,
        "minimum": minimum,
        "maximum": maximum,
        "options": options or [],
        "integer": integer,
        "helpText": help_text,
    }


def _feature(
    key: str,
    name: str,
    category: str,
    description: str,
    controls: list[Control] | None = None,
    *,
    live: bool = True,
    presets: dict[str, dict[str, Any]] | None = None,
) -> Feature:
    return {
        "key": key,
        "name": name,
        "category": category,
        "description": description,
        "controls": controls or [],
        "live": live,
        "presets": presets or {},
    }


FEATURES: list[Feature] = [
    _feature(
        "brightness_contrast",
        "Brightness & Contrast",
        "Basic",
        "Atur terang-gelap dan kontras citra.",
        [
            _control("brightness", "Brightness", default=0, minimum=-100, maximum=100),
            _control("contrast", "Contrast", default=0, minimum=-100, maximum=100),
        ],
        presets={
            "Ringan": {"brightness": 12, "contrast": 10},
            "Sedang": {"brightness": 25, "contrast": 25},
            "Kuat": {"brightness": 45, "contrast": 45},
        },
    ),
    _feature("grayscale", "RGB -> Grayscale", "Basic", "Konversi citra RGB ke grayscale."),
    _feature(
        "resize_percent",
        "Resize / Scaling",
        "Basic",
        "Ubah ukuran citra berdasarkan persentase lebar dan tinggi.",
        [
            _control("width_pct", "Width %", default=100, minimum=10, maximum=300),
            _control("height_pct", "Height %", default=100, minimum=10, maximum=300),
        ],
        live=False,
        presets={
            "Ringan": {"width_pct": 80, "height_pct": 80},
            "Sedang": {"width_pct": 150, "height_pct": 150},
            "Kuat": {"width_pct": 200, "height_pct": 200},
        },
    ),
    _feature("hist_equalization", "Histogram Equalization", "Enhancement", "Ratakan distribusi intensitas untuk meningkatkan kontras citra."),
    _feature(
        "sharpen",
        "Sharpening Fleksibel",
        "Enhancement",
        "Perjelas detail memakai unsharp masking.",
        [
            _control("amount", "Strength", default=1.0, minimum=0.1, maximum=4.0, integer=False),
            _control("kernel", "Kernel", default=5, minimum=3, maximum=21),
            _control("sigma", "Sigma", default=1.0, minimum=0.0, maximum=5.0, integer=False),
        ],
        presets={
            "Ringan": {"amount": 0.6, "kernel": 5, "sigma": 1.0},
            "Sedang": {"amount": 1.2, "kernel": 5, "sigma": 1.0},
            "Kuat": {"amount": 2.2, "kernel": 7, "sigma": 1.3},
        },
    ),
    _feature(
        "average_blur",
        "Smoothing / Average Blur",
        "Enhancement",
        "Haluskan citra memakai rata-rata kernel spasial.",
        [_control("kernel", "Kernel Size", default=5, minimum=1, maximum=31)],
        presets={"Ringan": {"kernel": 3}, "Sedang": {"kernel": 7}, "Kuat": {"kernel": 15}},
    ),
    _feature(
        "affine",
        "Rotate / Scale / Translate",
        "Transform",
        "Transformasi affine: rotasi, scaling, dan translasi.",
        [
            _control("angle", "Rotate deg", default=0, minimum=0, maximum=360),
            _control("scale_pct", "Scale %", default=100, minimum=10, maximum=250),
            _control("tx", "Translate X", default=0, minimum=-400, maximum=400),
            _control("ty", "Translate Y", default=0, minimum=-400, maximum=400),
        ],
        presets={
            "Ringan": {"angle": 10, "scale_pct": 100, "tx": 0, "ty": 0},
            "Sedang": {"angle": 45, "scale_pct": 110, "tx": 25, "ty": 25},
            "Kuat": {"angle": 120, "scale_pct": 90, "tx": 80, "ty": -40},
        },
    ),
    _feature("flip_horizontal", "Flip Horizontal", "Transform", "Balik citra kiri-kanan."),
    _feature("flip_vertical", "Flip Vertical", "Transform", "Balik citra atas-bawah."),
    _feature(
        "gaussian_blur",
        "Gaussian Blur",
        "Restoration",
        "Reduksi noise dengan Gaussian blur.",
        [
            _control("kernel", "Kernel Size", default=5, minimum=1, maximum=31),
            _control("sigma", "Sigma", default=0.0, minimum=0, maximum=8, integer=False),
        ],
        presets={
            "Ringan": {"kernel": 3, "sigma": 0.5},
            "Sedang": {"kernel": 7, "sigma": 1.3},
            "Kuat": {"kernel": 15, "sigma": 3.0},
        },
    ),
    _feature(
        "median_filter",
        "Median Filter",
        "Restoration",
        "Filter median untuk noise salt & pepper.",
        [_control("kernel", "Kernel Size", default=5, minimum=3, maximum=31)],
        presets={"Ringan": {"kernel": 3}, "Sedang": {"kernel": 5}, "Kuat": {"kernel": 11}},
    ),
    _feature(
        "noise_removal",
        "Salt & Pepper Removal",
        "Restoration",
        "Pembersihan noise impuls memakai median filter.",
        [_control("kernel", "Kernel Size", default=3, minimum=3, maximum=21)],
        presets={"Ringan": {"kernel": 3}, "Sedang": {"kernel": 5}, "Kuat": {"kernel": 9}},
    ),
    _feature(
        "threshold",
        "Threshold Binary",
        "Binary & Edge",
        "Ubah citra menjadi biner berdasarkan nilai threshold.",
        [
            _control("threshold", "Threshold", default=127, minimum=0, maximum=255),
            _control("invert", "Invert", kind="check", default=False),
        ],
        presets={"Ringan": {"threshold": 96}, "Sedang": {"threshold": 127}, "Kuat": {"threshold": 180}},
    ),
    _feature(
        "edge_detection",
        "Edge Detection Lengkap",
        "Binary & Edge",
        "Canny, Sobel, Prewitt, Robert, Laplacian, dan LoG.",
        [
            _control("method", "Metode", kind="combo", default="Canny", options=["Canny", "Sobel", "Prewitt", "Robert", "Laplacian", "Laplacian of Gaussian"]),
            _control("canny_low", "Canny Low", default=80, minimum=0, maximum=255),
            _control("canny_high", "Canny High", default=160, minimum=0, maximum=255),
            _control("kernel", "Kernel", default=3, minimum=3, maximum=15),
            _control("sigma", "LoG Sigma", default=0.0, minimum=0, maximum=5, integer=False),
        ],
        presets={
            "Ringan": {"canny_low": 60, "canny_high": 130, "kernel": 3},
            "Sedang": {"canny_low": 80, "canny_high": 160, "kernel": 3},
            "Kuat": {"canny_low": 120, "canny_high": 220, "kernel": 5},
        },
    ),
    _feature(
        "morphology",
        "Morphology Erosion/Dilation",
        "Binary & Edge",
        "Operasi morfologi dengan structuring element yang dapat diatur.",
        [
            _control("operation", "Operasi", kind="combo", default="erosion", options=["erosion", "dilation"]),
            _control("kernel", "Kernel", default=3, minimum=1, maximum=25),
            _control("iterations", "Iterations", default=1, minimum=1, maximum=10),
        ],
        presets={"Ringan": {"kernel": 3, "iterations": 1}, "Sedang": {"kernel": 5, "iterations": 2}, "Kuat": {"kernel": 9, "iterations": 4}},
    ),
    _feature("channel_split", "Channel Splitting RGB", "Color", "Tampilkan salah satu channel warna.", [_control("channel", "Channel", kind="combo", default="R", options=["R", "G", "B"])]),
    _feature(
        "hue_saturation",
        "Hue / Saturation",
        "Color",
        "Atur pergeseran hue dan saturasi warna citra.",
        [
            _control("hue", "Hue Shift", default=0, minimum=-90, maximum=90),
            _control("saturation", "Saturation", default=0, minimum=-100, maximum=100),
        ],
        presets={"Ringan": {"hue": 8, "saturation": 15}, "Sedang": {"hue": 22, "saturation": 35}, "Kuat": {"hue": 55, "saturation": 70}},
    ),
    _feature(
        "threshold_segmentation",
        "Threshold Segmentation",
        "Segmentation",
        "Segmentasi objek sederhana berdasarkan threshold intensitas.",
        [_control("threshold", "Threshold", default=127, minimum=0, maximum=255)],
        presets={"Ringan": {"threshold": 90}, "Sedang": {"threshold": 127}, "Kuat": {"threshold": 180}},
    ),
    _feature("edge_segmentation", "Edge-based Segmentation", "Segmentation", "Tandai tepi objek pada citra menggunakan hasil deteksi edge."),
    _feature(
        "region_segmentation",
        "Region-based / K-Means",
        "Segmentation",
        "Clustering warna sederhana untuk ekstraksi region.",
        [_control("k", "Jumlah Region K", default=3, minimum=2, maximum=32)],
        live=False,
        presets={"Ringan": {"k": 2}, "Sedang": {"k": 8}, "Kuat": {"k": 16}},
    ),
    _feature(
        "jpeg_simulation",
        "Simulasi JPEG Quality",
        "Compression",
        "Lihat dampak kualitas JPEG rendah sampai tinggi.",
        [_control("quality", "JPEG Quality", default=75, minimum=1, maximum=100)],
        presets={"Ringan": {"quality": 90}, "Sedang": {"quality": 60}, "Kuat": {"quality": 25}},
    ),
    _feature(
        "quantization",
        "Color Quantization",
        "Compression",
        "Simulasi kuantisasi warna untuk kompresi citra.",
        [_control("levels", "Levels", default=8, minimum=2, maximum=32)],
        presets={"Ringan": {"levels": 16}, "Sedang": {"levels": 8}, "Kuat": {"levels": 4}},
    ),
    _feature("rle_ratio", "RLE Compression Ratio", "Compression", "Hitung estimasi rasio kompresi RLE pada citra grayscale.", live=False),
    _feature("cnn_recognition", "CNN Object Recognition", "Machine Learning", "Klasifikasi objek opsional memakai model CNN ImageNet.", live=False),
]


def _as_int(params: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(round(float(params.get(key, default))))
    except (TypeError, ValueError):
        return default


def _as_float(params: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return default


def _as_bool(params: dict[str, Any], key: str, default: bool = False) -> bool:
    value = params.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _interpolation(params: dict[str, Any]) -> ip.InterpolationName:
    return "nearest" if params.get("interpolation") == "nearest" else "bilinear"


def _process_resize_percent(image: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    h, w = image.shape[:2]
    new_w = max(1, int(round(w * _as_float(params, "width_pct", 100.0) / 100.0)))
    new_h = max(1, int(round(h * _as_float(params, "height_pct", 100.0) / 100.0)))
    result = ip.resize_image(image, new_w, new_h, _interpolation(params))
    return result, f"Resize selesai: {new_w} x {new_h} px."


def _process_affine(image: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    result = ip.affine_transform(
        image,
        angle=_as_float(params, "angle", 0.0),
        scale=_as_float(params, "scale_pct", 100.0) / 100.0,
        translate_x=_as_int(params, "tx", 0),
        translate_y=_as_int(params, "ty", 0),
        interpolation=_interpolation(params),
    )
    return result, "Transformasi affine selesai."


def _process_crop(image: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    result = ip.crop(
        image,
        _as_int(params, "x1", 0),
        _as_int(params, "y1", 0),
        _as_int(params, "x2", image.shape[1]),
        _as_int(params, "y2", image.shape[0]),
    )
    return result, f"Crop selesai: {result.shape[1]} x {result.shape[0]} px."


def _process_rle_ratio(image: np.ndarray, _params: dict[str, Any]) -> tuple[np.ndarray, str]:
    ratio = ip.rle_compression_ratio(image)
    return image.copy(), f"Estimasi RLE: rasio original/RLE = {ratio:.3f}. Nilai > 1 berarti kompresi menguntungkan."


def _process_cnn_noop(image: np.ndarray, _params: dict[str, Any]) -> tuple[np.ndarray, str]:
    return image.copy(), "Gunakan tombol CNN di toolbar web untuk menjalankan klasifikasi objek."


PROCESSORS: dict[str, Processor] = {
    "brightness_contrast": lambda img, p: (ip.adjust_brightness_contrast(img, _as_int(p, "brightness"), _as_int(p, "contrast")), "Brightness & Contrast selesai."),
    "grayscale": lambda img, _p: (ip.rgb_to_grayscale(img), "Konversi grayscale selesai."),
    "resize_percent": _process_resize_percent,
    "hist_equalization": lambda img, _p: (ip.equalize_histogram(img), "Histogram equalization selesai."),
    "sharpen": lambda img, p: (ip.unsharp_mask(img, _as_float(p, "amount", 1.0), _as_int(p, "kernel", 5), _as_float(p, "sigma", 1.0)), "Sharpening selesai."),
    "average_blur": lambda img, p: (ip.average_smoothing(img, _as_int(p, "kernel", 5)), "Average blur selesai."),
    "affine": _process_affine,
    "flip_horizontal": lambda img, _p: (ip.flip_horizontal(img), "Flip horizontal selesai."),
    "flip_vertical": lambda img, _p: (ip.flip_vertical(img), "Flip vertical selesai."),
    "gaussian_blur": lambda img, p: (ip.gaussian_blur(img, _as_int(p, "kernel", 5), _as_float(p, "sigma", 0.0)), "Gaussian blur selesai."),
    "median_filter": lambda img, p: (ip.median_filter(img, _as_int(p, "kernel", 5)), "Median filter selesai."),
    "noise_removal": lambda img, p: (ip.remove_salt_pepper(img, _as_int(p, "kernel", 3)), "Salt & pepper removal selesai."),
    "threshold": lambda img, p: (ip.threshold_binary(img, _as_int(p, "threshold", 127), _as_bool(p, "invert")), "Threshold binary selesai."),
    "edge_detection": lambda img, p: (ip.edge_detection(img, str(p.get("method", "Canny")), _as_int(p, "canny_low", 80), _as_int(p, "canny_high", 160), _as_int(p, "kernel", 3), _as_float(p, "sigma", 0.0)), "Edge detection selesai."),
    "morphology": lambda img, p: (ip.morphology(img, cast(Literal["erosion", "dilation"], p.get("operation", "erosion")), _as_int(p, "kernel", 3), _as_int(p, "iterations", 1)), "Morphology selesai."),
    "channel_split": lambda img, p: (ip.split_channel(img, cast(Literal["R", "G", "B"], p.get("channel", "R"))), "Channel splitting selesai."),
    "hue_saturation": lambda img, p: (ip.adjust_hue_saturation(img, _as_int(p, "hue"), _as_int(p, "saturation")), "Hue / Saturation selesai."),
    "threshold_segmentation": lambda img, p: (ip.threshold_segmentation(img, _as_int(p, "threshold", 127)), "Threshold segmentation selesai."),
    "edge_segmentation": lambda img, _p: (ip.edge_based_segmentation(img), "Edge-based segmentation selesai."),
    "region_segmentation": lambda img, p: (ip.region_based_segmentation(img, _as_int(p, "k", 3)), "Region-based segmentation selesai."),
    "jpeg_simulation": lambda img, p: (ip.simulate_jpeg(img, _as_int(p, "quality", 75)), "Simulasi JPEG selesai."),
    "quantization": lambda img, p: (ip.quantize_colors(img, _as_int(p, "levels", 8)), "Color quantization selesai."),
    "rle_ratio": _process_rle_ratio,
    "cnn_recognition": _process_cnn_noop,
    "crop": _process_crop,
}


def _parse_params(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        params = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Parameter bukan JSON valid.") from exc
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="Parameter harus berupa object JSON.")
    return params


async def _read_image(upload: UploadFile) -> np.ndarray:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="File gambar kosong.")
    try:
        with Image.open(io.BytesIO(data)) as pil:
            pil.load()
            return np.asarray(pil.convert("RGB"), dtype=np.uint8).copy()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Gagal membaca gambar: {exc}") from exc


def _image_response(image: np.ndarray, message: str, *, image_format: str = "PNG", quality: int = 90) -> Response:
    payload, media_type = _array_to_bytes(image, image_format=image_format, quality=quality)
    return Response(
        payload,
        media_type=media_type,
        headers={
            "X-Image-Message": quote(message),
            "X-Image-Width": str(image.shape[1]),
            "X-Image-Height": str(image.shape[0]),
        },
    )


def _array_to_bytes(image: np.ndarray, *, image_format: str = "PNG", quality: int = 90) -> tuple[bytes, str]:
    fmt = image_format.upper().replace("JPG", "JPEG")
    if fmt not in {"PNG", "JPEG", "BMP", "TIFF"}:
        raise HTTPException(status_code=400, detail="Format output harus PNG, JPEG, BMP, atau TIFF.")
    pil = Image.fromarray(ip.ensure_uint8(image)).convert("RGB")
    stream = io.BytesIO()
    if fmt == "JPEG":
        pil.save(stream, format=fmt, quality=int(np.clip(quality, 1, 100)), optimize=True)
        return stream.getvalue(), "image/jpeg"
    pil.save(stream, format=fmt)
    media_type = {
        "PNG": "image/png",
        "BMP": "image/bmp",
        "TIFF": "image/tiff",
    }[fmt]
    return stream.getvalue(), media_type


@app.get("/")
def index() -> FileResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend belum tersedia.")
    return FileResponse(index_file)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/features")
def features() -> dict[str, Any]:
    categories = []
    for feature in FEATURES:
        if feature["category"] not in categories:
            categories.append(feature["category"])
    return {"categories": categories, "features": FEATURES, "models": list(MODEL_NAMES), "defaultModel": DEFAULT_MODEL_NAME}


@app.post("/api/process")
async def process_image(
    image: UploadFile = File(...),
    feature: str = Form(...),
    params: str = Form("{}"),
) -> Response:
    if feature not in PROCESSORS:
        raise HTTPException(status_code=404, detail=f"Fitur tidak dikenal: {feature}")
    arr = await _read_image(image)
    parsed = _parse_params(params)
    try:
        result, message = PROCESSORS[feature](arr, parsed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Gagal memproses {feature}: {exc}") from exc
    return _image_response(result, message)


@app.post("/api/export")
async def export_image(
    image: UploadFile = File(...),
    image_format: str = Form("PNG"),
    quality: int = Form(90),
) -> Response:
    arr = await _read_image(image)
    payload, media_type = _array_to_bytes(arr, image_format=image_format, quality=quality)
    suffix = image_format.lower().replace("jpeg", "jpg")
    return Response(
        payload,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="mini-photoshop-result.{suffix}"'},
    )


@app.post("/api/histogram")
async def histogram(image: UploadFile = File(...)) -> JSONResponse:
    arr = await _read_image(image)
    hist = ip.compute_histograms(arr)
    as_lists = {name: values.astype(float).round(2).tolist() for name, values in hist.items()}
    return JSONResponse({"width": int(arr.shape[1]), "height": int(arr.shape[0]), "histograms": as_lists})


@app.post("/api/cnn")
async def cnn_recognition(
    image: UploadFile = File(...),
    model_name: str = Form(DEFAULT_MODEL_NAME),
    top_k: int = Form(5),
) -> JSONResponse:
    arr = await _read_image(image)
    try:
        from mini_photoshop.ml import CNNRecognizer

        recognizer = CNNRecognizer(model_name=model_name)
        predictions = recognizer.predict(arr, top_k=top_k)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="CNN membutuhkan TensorFlow dan bobot ImageNet. Install requirements-ml.txt bila ingin memakai fitur ini. Detail: " + str(exc),
        ) from exc
    return JSONResponse(
        {
            "model": model_name,
            "predictions": [
                {"label": item.label, "confidence": item.confidence}
                for item in predictions
            ],
        }
    )
