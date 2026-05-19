"""Modern Tkinter GUI for the Mini Photoshop digital image processing project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, cast

import numpy as np
from PIL import Image, ImageFile, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from . import image_processor as ip
from .config import (
    ACCENT,
    APP_TITLE,
    CANVAS_BG,
    DEFAULT_WINDOW_SIZE,
    FEATURE_CATEGORIES,
    HISTORY_LIMIT,
    MIN_WINDOW_SIZE,
    MUTED_TEXT,
    PANEL_BG,
    PRESETS,
)
from .history import ImageHistory
from .ml import DEFAULT_MODEL_NAME, MODEL_NAMES
from .ui_components import ScrollableFrame, make_action_button, make_slider

FeatureProcessor = Callable[[np.ndarray, dict[str, Any]], np.ndarray | tuple[np.ndarray, str]]

# Local coursework images can be very large (for example panoramic or scanned
# images). Pillow protects against decompression-bomb attacks by default and may
# reject files above its pixel threshold. This desktop app is intentionally used
# on local, user-selected images, so the original pixel dimensions are allowed.
# The preview canvas is resized only for display; processing and saving still use
# the full-resolution NumPy array.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

LARGE_IMAGE_UNDO_PIXEL_THRESHOLD = 50_000_000
LARGE_IMAGE_UNDO_LIMIT = 3



def prediction_to_label_score(prediction: Any) -> tuple[str, float]:
    """Normalize CNN prediction objects/tuples into (label, confidence).

    CNNRecognizer returns Prediction(label, confidence), while some Keras
    helpers return tuples like (class_id, label, score). The GUI accepts both
    forms so the ML button does not crash when implementation details change.
    """
    label = getattr(prediction, "label", None)
    confidence = getattr(prediction, "confidence", None)
    if confidence is None:
        confidence = getattr(prediction, "score", None)
    if confidence is None:
        confidence = getattr(prediction, "probability", None)
    if label is not None and confidence is not None:
        return str(label), float(confidence)

    if isinstance(prediction, (tuple, list)):
        if len(prediction) >= 3 and isinstance(prediction[1], str):
            return str(prediction[1]).replace("_", " "), float(prediction[2])
        if len(prediction) >= 2:
            return str(prediction[0]).replace("_", " "), float(prediction[1])

    raise TypeError(f"Format prediksi CNN tidak didukung: {type(prediction).__name__}")


def format_predictions(predictions: list[Any]) -> str:
    """Build a readable numbered prediction list for the message box."""
    if not predictions:
        return "Tidak ada objek yang dikenali."
    rows = []
    for idx, prediction in enumerate(predictions):
        label, confidence = prediction_to_label_score(prediction)
        rows.append(f"{idx + 1}. {label}: {confidence:.2%}")
    return "\n".join(rows)


@dataclass(frozen=True)
class ControlSpec:
    key: str
    label: str
    kind: str = "slider"  # slider, combo, check
    default: Any = 0
    minimum: float = 0
    maximum: float = 100
    options: tuple[str, ...] = ()
    integer: bool = True
    help_text: str = ""


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    name: str
    category: str
    description: str
    controls: tuple[ControlSpec, ...]
    processor: FeatureProcessor
    live: bool = True
    presets: dict[str, dict[str, Any]] = field(default_factory=dict)


class MiniPhotoshopApp(tk.Tk):
    """Desktop editor with flexible live parameters and before-after preview."""

    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(DEFAULT_WINDOW_SIZE)
        self.minsize(*MIN_WINDOW_SIZE)

        self.original_image: Optional[np.ndarray] = None
        self.base_image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None
        self.current_path: Optional[Path] = None
        self.history = ImageHistory(limit=HISTORY_LIMIT)

        self._before_photo: Optional[ImageTk.PhotoImage] = None
        self._after_photo: Optional[ImageTk.PhotoImage] = None
        self._before_scale = 1.0
        self._after_scale = 1.0
        self._after_offset = (0, 0)
        self._crop_start_canvas: Optional[tuple[int, int]] = None
        self._crop_end_canvas: Optional[tuple[int, int]] = None
        self._crop_rect_id: Optional[int] = None
        self._update_job: Optional[str] = None
        self._cnn_recognizers: dict[str, Any] = {}

        self.features = self._build_features()
        self.feature_by_key = {feature.key: feature for feature in self.features}
        self.active_feature_key = tk.StringVar(value=self.features[0].key)
        self.param_vars: dict[str, tk.Variable] = {}

        self.live_preview_var = tk.BooleanVar(value=True)
        self.show_original_before_var = tk.BooleanVar(value=True)
        self.preset_var = tk.StringVar(value="Manual")
        self.interpolation_var = tk.StringVar(value="bilinear")
        self.save_quality_var = tk.IntVar(value=90)
        self.cnn_model_var = tk.StringVar(value=DEFAULT_MODEL_NAME)
        self.cnn_top_k_var = tk.IntVar(value=5)
        self.status_var = tk.StringVar(value="Buka gambar untuk mulai mengedit.")
        self.info_var = tk.StringVar(value="Belum ada gambar.")

        self._configure_style()
        self._build_menu()
        self._build_ui()
        self._select_feature(self.features[0].key)
        self.after(80, self._draw_empty_canvases)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=PANEL_BG)
        style.configure("Card.TFrame", background="#172033")
        style.configure("TLabel", background=PANEL_BG, foreground="#e5e7eb")
        style.configure("Muted.TLabel", background=PANEL_BG, foreground=MUTED_TEXT)
        style.configure("Title.TLabel", background=PANEL_BG, foreground="#f8fafc", font=("Segoe UI", 14, "bold"))
        style.configure("Subtitle.TLabel", background=PANEL_BG, foreground=ACCENT, font=("Segoe UI", 10, "bold"))
        style.configure("TButton", padding=7, font=("Segoe UI", 9))
        style.configure("Accent.TButton", padding=8, font=("Segoe UI", 9, "bold"))
        style.configure("TCheckbutton", background=PANEL_BG, foreground="#e5e7eb")
        style.configure("TLabelframe", background=PANEL_BG, foreground="#e5e7eb")
        style.configure("TLabelframe.Label", background=PANEL_BG, foreground="#f8fafc", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", background="#0f172a", fieldbackground="#0f172a", foreground="#e5e7eb", rowheight=27, borderwidth=0)
        style.configure("Treeview.Heading", background="#1f2937", foreground="#e5e7eb")
        style.map("Treeview", background=[("selected", "#075985")], foreground=[("selected", "#ffffff")])

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Load Image", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Result", command=self.save_image, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As JPEG Quality...", command=self.save_with_quality)
        file_menu.add_separator()
        file_menu.add_command(label="Reset ke Gambar Awal", command=self.reset_to_original)
        file_menu.add_command(label="Keluar", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=False)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Apply Efek Aktif", command=self.apply_active_feature, accelerator="Ctrl+Enter")
        edit_menu.add_command(label="Cancel Preview", command=self.cancel_preview)
        edit_menu.add_command(label="Reset Parameter", command=self.reset_parameters)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        view_menu = tk.Menu(menubar, tearoff=False)
        view_menu.add_checkbutton(label="Live Preview", variable=self.live_preview_var, command=self.schedule_preview)
        view_menu.add_checkbutton(label="Before selalu gambar awal", variable=self.show_original_before_var, command=self.refresh_canvases)
        view_menu.add_command(label="Histogram Before vs After", command=self.show_histogram)
        menubar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="Panduan Singkat", command=self.show_help)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)
        self.bind_all("<Control-o>", lambda _event: self.open_image())
        self.bind_all("<Control-s>", lambda _event: self.save_image())
        self.bind_all("<Control-z>", lambda _event: self.undo())
        self.bind_all("<Control-y>", lambda _event: self.redo())
        self.bind_all("<Control-Return>", lambda _event: self.apply_active_feature())

    def _build_ui(self) -> None:
        shell = ttk.Frame(self, padding=10)
        shell.pack(fill=tk.BOTH, expand=True)

        self._build_toolbar(shell)

        body = ttk.PanedWindow(shell, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.sidebar = ttk.Frame(body, width=260)
        body.add(self.sidebar, weight=0)
        self._build_sidebar(self.sidebar)

        self.preview_area = ttk.Frame(body)
        body.add(self.preview_area, weight=3)
        self._build_preview_area(self.preview_area)

        self.param_panel = ttk.Frame(body, width=360)
        body.add(self.param_panel, weight=0)
        self._build_parameter_panel(self.param_panel)

        self._build_statusbar(shell)

    def _build_toolbar(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="Mini Photoshop Pro", style="Title.TLabel").pack(side=tk.LEFT, padx=(0, 16))
        ttk.Button(toolbar, text="📂 Load", command=self.open_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="💾 Save", command=self.save_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="↩ Undo", command=self.undo).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="↪ Redo", command=self.redo).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="⟲ Reset", command=self.reset_to_original).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="📊 Histogram", command=self.show_histogram).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="🧠 CNN", command=self.recognize_object).pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(toolbar, text="Live Preview", variable=self.live_preview_var, command=self.schedule_preview).pack(side=tk.RIGHT, padx=8)
        ttk.Checkbutton(toolbar, text="Before = Original", variable=self.show_original_before_var, command=self.refresh_canvases).pack(side=tk.RIGHT, padx=8)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Kelompok Fitur", style="Subtitle.TLabel").pack(anchor="w", pady=(0, 6))
        self.feature_tree = ttk.Treeview(parent, show="tree", selectmode="browse", height=25)
        self.feature_tree.pack(fill=tk.BOTH, expand=True)
        for category in FEATURE_CATEGORIES:
            cat_id = f"cat::{category}"
            self.feature_tree.insert("", "end", iid=cat_id, text=category, open=True)
            for feature in [f for f in self.features if f.category == category]:
                self.feature_tree.insert(cat_id, "end", iid=feature.key, text=feature.name)
        self.feature_tree.bind("<<TreeviewSelect>>", self._on_feature_selected)
        self.feature_tree.selection_set(self.active_feature_key.get())

        quick = ttk.LabelFrame(parent, text="Aksi Cepat", padding=8)
        quick.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(quick, text="Grayscale", command=lambda: self.quick_apply("grayscale")).pack(fill=tk.X, pady=3)
        ttk.Button(quick, text="Sharpen", command=lambda: self.quick_apply("sharpen")).pack(fill=tk.X, pady=3)
        ttk.Button(quick, text="Canny Edge", command=lambda: self.quick_apply("edge_detection", {"method": "Canny"})).pack(fill=tk.X, pady=3)
        ttk.Button(quick, text="Equalize Histogram", command=lambda: self.quick_apply("hist_equalization")).pack(fill=tk.X, pady=3)

    def _build_preview_area(self, parent: ttk.Frame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        ttk.Label(parent, textvariable=self.info_var, style="Muted.TLabel").grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))

        before_frame = ttk.LabelFrame(parent, text="Before", padding=6)
        before_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        before_frame.grid_rowconfigure(0, weight=1)
        before_frame.grid_columnconfigure(0, weight=1)
        self.before_canvas = tk.Canvas(before_frame, bg=CANVAS_BG, highlightthickness=0)
        self.before_canvas.grid(row=0, column=0, sticky="nsew")
        self.before_canvas.bind("<Configure>", lambda _event: self.refresh_canvases())

        after_frame = ttk.LabelFrame(parent, text="After / Live Result", padding=6)
        after_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        after_frame.grid_rowconfigure(0, weight=1)
        after_frame.grid_columnconfigure(0, weight=1)
        self.after_canvas = tk.Canvas(after_frame, bg=CANVAS_BG, highlightthickness=0, cursor="crosshair")
        self.after_canvas.grid(row=0, column=0, sticky="nsew")
        self.after_canvas.bind("<Configure>", lambda _event: self.refresh_canvases())
        self.after_canvas.bind("<ButtonPress-1>", self._start_crop_drag)
        self.after_canvas.bind("<B1-Motion>", self._drag_crop)
        self.after_canvas.bind("<ButtonRelease-1>", self._finish_crop_drag)

    def _build_parameter_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Parameter Fleksibel", style="Subtitle.TLabel").pack(anchor="w", pady=(0, 6))
        self.feature_title_var = tk.StringVar(value="")
        self.feature_desc_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.feature_title_var, style="Title.TLabel", wraplength=330).pack(anchor="w")
        ttk.Label(parent, textvariable=self.feature_desc_var, style="Muted.TLabel", wraplength=330, justify="left").pack(anchor="w", pady=(2, 10))

        preset_row = ttk.Frame(parent)
        preset_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(preset_row, text="Preset").pack(side=tk.LEFT)
        preset_values = ["Manual", *PRESETS.keys()]
        self.preset_combo = ttk.Combobox(preset_row, textvariable=self.preset_var, values=preset_values, state="readonly", width=16)
        self.preset_combo.pack(side=tk.RIGHT)
        self.preset_combo.bind("<<ComboboxSelected>>", lambda _event: self.apply_preset())

        interpolation_row = ttk.Frame(parent)
        interpolation_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(interpolation_row, text="Interpolasi Transform").pack(side=tk.LEFT)
        self.interpolation_combo = ttk.Combobox(interpolation_row, textvariable=self.interpolation_var, values=("nearest", "bilinear"), state="readonly", width=16)
        self.interpolation_combo.pack(side=tk.RIGHT)
        self.interpolation_combo.bind("<<ComboboxSelected>>", lambda _event: self.schedule_preview())

        cnn_box = ttk.LabelFrame(parent, text="CNN Recognition", padding=8)
        cnn_box.pack(fill=tk.X, pady=(0, 8))
        cnn_box.grid_columnconfigure(1, weight=1)
        ttk.Label(cnn_box, text="Model").grid(row=0, column=0, sticky="w", pady=3)
        self.cnn_model_combo = ttk.Combobox(cnn_box, textvariable=self.cnn_model_var, values=MODEL_NAMES, state="readonly", width=18)
        self.cnn_model_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=3)
        ttk.Label(cnn_box, text="Top-K").grid(row=1, column=0, sticky="w", pady=3)
        self.cnn_top_k_spin = ttk.Spinbox(cnn_box, from_=1, to=10, textvariable=self.cnn_top_k_var, width=6)
        self.cnn_top_k_spin.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=3)
        ttk.Button(cnn_box, text="Jalankan CNN", command=self.recognize_object).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        self.controls_scroll = ScrollableFrame(parent)
        self.controls_scroll.pack(fill=tk.BOTH, expand=True)

        actions = ttk.Frame(parent)
        actions.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(actions, text="Preview", command=self.preview_active_feature).grid(row=0, column=0, sticky="ew", padx=3, pady=3)
        ttk.Button(actions, text="Apply", style="Accent.TButton", command=self.apply_active_feature).grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        ttk.Button(actions, text="Cancel", command=self.cancel_preview).grid(row=0, column=2, sticky="ew", padx=3, pady=3)
        ttk.Button(actions, text="Reset Param", command=self.reset_parameters).grid(row=1, column=0, sticky="ew", padx=3, pady=3)
        ttk.Button(actions, text="Crop Area", command=self.crop_selection).grid(row=1, column=1, sticky="ew", padx=3, pady=3)
        ttk.Button(actions, text="Histogram", command=self.show_histogram).grid(row=1, column=2, sticky="ew", padx=3, pady=3)
        for col in range(3):
            actions.grid_columnconfigure(col, weight=1)

    def _build_statusbar(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(bar, textvariable=self.status_var, style="Muted.TLabel", anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(bar, text="Ctrl+O Load  •  Ctrl+S Save  •  Ctrl+Enter Apply", style="Muted.TLabel").pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Features and processors
    # ------------------------------------------------------------------
    def _build_features(self) -> list[FeatureSpec]:
        C = ControlSpec
        return [
            FeatureSpec(
                "brightness_contrast",
                "Brightness & Contrast",
                "Basic",
                "Atur terang-gelap dan kontras secara realtime dengan slider.",
                (C("brightness", "Brightness", default=0, minimum=-100, maximum=100), C("contrast", "Contrast", default=0, minimum=-100, maximum=100)),
                lambda img, p: ip.adjust_brightness_contrast(img, int(p["brightness"]), int(p["contrast"])),
                presets={"Ringan": {"brightness": 12, "contrast": 10}, "Sedang": {"brightness": 25, "contrast": 25}, "Kuat": {"brightness": 45, "contrast": 45}},
            ),
            FeatureSpec("grayscale", "RGB → Grayscale", "Basic", "Konversi citra RGB ke grayscale.", (), lambda img, _p: ip.rgb_to_grayscale(img)),
            FeatureSpec(
                "resize_percent",
                "Resize / Scaling",
                "Basic",
                "Ubah ukuran citra berdasarkan persentase lebar dan tinggi.",
                (C("width_pct", "Width %", default=100, minimum=10, maximum=300), C("height_pct", "Height %", default=100, minimum=10, maximum=300)),
                self._process_resize_percent,
                live=False,
                presets={"Ringan": {"width_pct": 80, "height_pct": 80}, "Sedang": {"width_pct": 150, "height_pct": 150}, "Kuat": {"width_pct": 200, "height_pct": 200}},
            ),
            FeatureSpec("hist_equalization", "Histogram Equalization", "Enhancement", "Ratakan distribusi intensitas untuk meningkatkan kontras citra.", (), lambda img, _p: ip.equalize_histogram(img)),
            FeatureSpec(
                "sharpen",
                "Sharpening Fleksibel",
                "Enhancement",
                "Perjelas detail memakai unsharp masking dengan kekuatan yang bisa diatur.",
                (C("amount", "Strength", default=1.0, minimum=0.1, maximum=4.0, integer=False), C("kernel", "Kernel", default=5, minimum=3, maximum=21), C("sigma", "Sigma", default=1.0, minimum=0.0, maximum=5.0, integer=False)),
                lambda img, p: ip.unsharp_mask(img, float(p["amount"]), int(p["kernel"]), float(p["sigma"])),
                presets={"Ringan": {"amount": 0.6, "kernel": 5, "sigma": 1.0}, "Sedang": {"amount": 1.2, "kernel": 5, "sigma": 1.0}, "Kuat": {"amount": 2.2, "kernel": 7, "sigma": 1.3}},
            ),
            FeatureSpec(
                "average_blur",
                "Smoothing / Average Blur",
                "Enhancement",
                "Haluskan citra memakai rata-rata kernel spasial.",
                (C("kernel", "Kernel Size", default=5, minimum=1, maximum=31),),
                lambda img, p: ip.average_smoothing(img, int(p["kernel"])),
                presets={"Ringan": {"kernel": 3}, "Sedang": {"kernel": 7}, "Kuat": {"kernel": 15}},
            ),
            FeatureSpec(
                "affine",
                "Rotate / Scale / Translate",
                "Transform",
                "Transformasi affine: rotasi, scaling, dan translasi dengan pilihan interpolasi.",
                (C("angle", "Rotate °", default=0, minimum=0, maximum=360), C("scale_pct", "Scale %", default=100, minimum=10, maximum=250), C("tx", "Translate X", default=0, minimum=-400, maximum=400), C("ty", "Translate Y", default=0, minimum=-400, maximum=400)),
                self._process_affine,
                presets={"Ringan": {"angle": 10, "scale_pct": 100, "tx": 0, "ty": 0}, "Sedang": {"angle": 45, "scale_pct": 110, "tx": 25, "ty": 25}, "Kuat": {"angle": 120, "scale_pct": 90, "tx": 80, "ty": -40}},
            ),
            FeatureSpec("flip_horizontal", "Flip Horizontal", "Transform", "Balik citra kiri-kanan.", (), lambda img, _p: ip.flip_horizontal(img)),
            FeatureSpec("flip_vertical", "Flip Vertical", "Transform", "Balik citra atas-bawah.", (), lambda img, _p: ip.flip_vertical(img)),
            FeatureSpec(
                "gaussian_blur",
                "Gaussian Blur",
                "Restoration",
                "Reduksi noise dengan Gaussian blur, kernel dan sigma dapat diatur.",
                (C("kernel", "Kernel Size", default=5, minimum=1, maximum=31), C("sigma", "Sigma", default=0.0, minimum=0, maximum=8, integer=False)),
                lambda img, p: ip.gaussian_blur(img, int(p["kernel"]), float(p["sigma"])),
                presets={"Ringan": {"kernel": 3, "sigma": 0.5}, "Sedang": {"kernel": 7, "sigma": 1.3}, "Kuat": {"kernel": 15, "sigma": 3.0}},
            ),
            FeatureSpec(
                "median_filter",
                "Median Filter",
                "Restoration",
                "Filter median untuk noise salt & pepper.",
                (C("kernel", "Kernel Size", default=5, minimum=3, maximum=31),),
                lambda img, p: ip.median_filter(img, int(p["kernel"])),
                presets={"Ringan": {"kernel": 3}, "Sedang": {"kernel": 5}, "Kuat": {"kernel": 11}},
            ),
            FeatureSpec(
                "noise_removal",
                "Salt & Pepper Removal",
                "Restoration",
                "Pembersihan noise impuls memakai median filter.",
                (C("kernel", "Kernel Size", default=3, minimum=3, maximum=21),),
                lambda img, p: ip.remove_salt_pepper(img, int(p["kernel"])),
                presets={"Ringan": {"kernel": 3}, "Sedang": {"kernel": 5}, "Kuat": {"kernel": 9}},
            ),
            FeatureSpec(
                "threshold",
                "Threshold Binary",
                "Binary & Edge",
                "Ubah citra menjadi biner berdasarkan nilai threshold.",
                (C("threshold", "Threshold", default=127, minimum=0, maximum=255), C("invert", "Invert", kind="check", default=False)),
                lambda img, p: ip.threshold_binary(img, int(p["threshold"]), bool(p["invert"])),
                presets={"Ringan": {"threshold": 96}, "Sedang": {"threshold": 127}, "Kuat": {"threshold": 180}},
            ),
            FeatureSpec(
                "edge_detection",
                "Edge Detection Lengkap",
                "Binary & Edge",
                "Canny, Sobel, Prewitt, Robert, Laplacian, dan LoG dengan parameter fleksibel.",
                (
                    C("method", "Metode", kind="combo", default="Canny", options=("Canny", "Sobel", "Prewitt", "Robert", "Laplacian", "Laplacian of Gaussian")),
                    C("canny_low", "Canny Low", default=80, minimum=0, maximum=255),
                    C("canny_high", "Canny High", default=160, minimum=0, maximum=255),
                    C("kernel", "Kernel", default=3, minimum=3, maximum=15),
                    C("sigma", "LoG Sigma", default=0.0, minimum=0, maximum=5, integer=False),
                ),
                lambda img, p: ip.edge_detection(img, str(p["method"]), int(p["canny_low"]), int(p["canny_high"]), int(p["kernel"]), float(p["sigma"])),
                presets={"Ringan": {"canny_low": 60, "canny_high": 130, "kernel": 3}, "Sedang": {"canny_low": 80, "canny_high": 160, "kernel": 3}, "Kuat": {"canny_low": 120, "canny_high": 220, "kernel": 5}},
            ),
            FeatureSpec(
                "morphology",
                "Morphology Erosion/Dilation",
                "Binary & Edge",
                "Operasi morfologi biner dengan structuring element yang dapat diatur.",
                (C("operation", "Operasi", kind="combo", default="erosion", options=("erosion", "dilation")), C("kernel", "Kernel", default=3, minimum=1, maximum=25), C("iterations", "Iterations", default=1, minimum=1, maximum=10)),
                lambda img, p: ip.morphology(img, cast(Literal["erosion", "dilation"], p["operation"]), int(p["kernel"]), int(p["iterations"])),
                presets={"Ringan": {"kernel": 3, "iterations": 1}, "Sedang": {"kernel": 5, "iterations": 2}, "Kuat": {"kernel": 9, "iterations": 4}},
            ),
            FeatureSpec(
                "channel_split",
                "Channel Splitting RGB",
                "Color",
                "Tampilkan salah satu channel warna R, G, atau B.",
                (C("channel", "Channel", kind="combo", default="R", options=("R", "G", "B")),),
                lambda img, p: ip.split_channel(img, cast(Literal["R", "G", "B"], p["channel"])),
            ),
            FeatureSpec(
                "hue_saturation",
                "Hue / Saturation",
                "Color",
                "Atur pergeseran hue dan saturasi warna citra.",
                (C("hue", "Hue Shift", default=0, minimum=-90, maximum=90), C("saturation", "Saturation", default=0, minimum=-100, maximum=100)),
                lambda img, p: ip.adjust_hue_saturation(img, int(p["hue"]), int(p["saturation"])),
                presets={"Ringan": {"hue": 8, "saturation": 15}, "Sedang": {"hue": 22, "saturation": 35}, "Kuat": {"hue": 55, "saturation": 70}},
            ),
            FeatureSpec(
                "threshold_segmentation",
                "Threshold Segmentation",
                "Segmentation",
                "Segmentasi objek sederhana berdasarkan threshold intensitas.",
                (C("threshold", "Threshold", default=127, minimum=0, maximum=255),),
                lambda img, p: ip.threshold_segmentation(img, int(p["threshold"])),
                presets={"Ringan": {"threshold": 90}, "Sedang": {"threshold": 127}, "Kuat": {"threshold": 180}},
            ),
            FeatureSpec("edge_segmentation", "Edge-based Segmentation", "Segmentation", "Tandai tepi objek pada citra menggunakan hasil deteksi edge.", (), lambda img, _p: ip.edge_based_segmentation(img)),
            FeatureSpec(
                "region_segmentation",
                "Region-based / K-Means",
                "Segmentation",
                "Clustering warna sederhana untuk ekstraksi region.",
                (C("k", "Jumlah Region K", default=3, minimum=2, maximum=32),),
                lambda img, p: ip.region_based_segmentation(img, int(p["k"])),
                live=False,
                presets={"Ringan": {"k": 2}, "Sedang": {"k": 8}, "Kuat": {"k": 16}},
            ),
            FeatureSpec(
                "jpeg_simulation",
                "Simulasi JPEG Quality",
                "Compression",
                "Lihat dampak kualitas JPEG rendah sampai tinggi.",
                (C("quality", "JPEG Quality", default=75, minimum=1, maximum=100),),
                lambda img, p: ip.simulate_jpeg(img, int(p["quality"])),
                presets={"Ringan": {"quality": 90}, "Sedang": {"quality": 60}, "Kuat": {"quality": 25}},
            ),
            FeatureSpec(
                "quantization",
                "Color Quantization",
                "Compression",
                "Simulasi kuantisasi warna untuk kompresi citra.",
                (C("levels", "Levels", default=8, minimum=2, maximum=32),),
                lambda img, p: ip.quantize_colors(img, int(p["levels"])),
                presets={"Ringan": {"levels": 16}, "Sedang": {"levels": 8}, "Kuat": {"levels": 4}},
            ),
            FeatureSpec(
                "rle_ratio",
                "RLE Compression Ratio",
                "Compression",
                "Hitung estimasi rasio kompresi RLE pada citra grayscale.",
                (),
                self._process_rle_ratio,
                live=False,
            ),
            FeatureSpec(
                "cnn_recognition",
                "CNN Object Recognition",
                "Machine Learning",
                "Klasifikasi objek opsional memakai pilihan model CNN ImageNet: MobileNetV2, ResNet50, EfficientNetB0, atau InceptionV3.",
                (),
                self._process_cnn_placeholder,
                live=False,
            ),
        ]

    def _process_resize_percent(self, image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        h, w = image.shape[:2]
        new_w = max(1, int(round(w * float(params["width_pct"]) / 100.0)))
        new_h = max(1, int(round(h * float(params["height_pct"]) / 100.0)))
        return ip.resize_image(image, new_w, new_h, self._selected_interpolation())

    def _process_affine(self, image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        return ip.affine_transform(
            image,
            angle=float(params["angle"]),
            scale=float(params["scale_pct"]) / 100.0,
            translate_x=int(params["tx"]),
            translate_y=int(params["ty"]),
            interpolation=self._selected_interpolation(),
        )

    def _selected_interpolation(self) -> ip.InterpolationName:
        interpolation = self.interpolation_var.get()
        if interpolation not in ("nearest", "bilinear"):
            return "bilinear"
        return cast(ip.InterpolationName, interpolation)

    def _process_rle_ratio(self, image: np.ndarray, _params: dict[str, Any]) -> tuple[np.ndarray, str]:
        ratio = ip.rle_compression_ratio(image)
        return image.copy(), f"Estimasi RLE: rasio original/RLE = {ratio:.3f}. Nilai > 1 berarti kompresi menguntungkan."

    def _process_cnn_placeholder(self, image: np.ndarray, _params: dict[str, Any]) -> tuple[np.ndarray, str]:
        self.recognize_object()
        return image, f"CNN dijalankan dengan model {self.cnn_model_var.get()}."

    # ------------------------------------------------------------------
    # Feature panel behavior
    # ------------------------------------------------------------------
    def _on_feature_selected(self, _event: tk.Event) -> None:
        selection = self.feature_tree.selection()
        if not selection:
            return
        key = selection[0]
        if key.startswith("cat::"):
            return
        self._select_feature(key)

    def _select_feature(self, key: str) -> None:
        if key not in self.feature_by_key:
            return
        self.active_feature_key.set(key)
        feature = self.feature_by_key[key]
        self.feature_title_var.set(feature.name)
        self.feature_desc_var.set(feature.description)
        self.preset_var.set("Manual")
        self._rebuild_controls(feature)
        self.schedule_preview()

    def _rebuild_controls(self, feature: FeatureSpec) -> None:
        for child in self.controls_scroll.body.winfo_children():
            child.destroy()
        self.param_vars.clear()

        if not feature.controls:
            ttk.Label(self.controls_scroll.body, text="Fitur ini tidak membutuhkan parameter tambahan.", style="Muted.TLabel", wraplength=300).grid(row=0, column=0, sticky="ew", pady=8)
            return

        for row, control in enumerate(feature.controls):
            if control.kind == "slider":
                # ttk.Scale internally works with floating point values. Store sliders
                # as DoubleVar and cast to int later when the feature requires it.
                var: tk.Variable = tk.DoubleVar(value=float(control.default))
                self.param_vars[control.key] = var
                make_slider(self.controls_scroll.body, control.label, var, control.minimum, control.maximum, row, self._on_parameter_changed)
            elif control.kind == "combo":
                var = tk.StringVar(value=str(control.default))
                self.param_vars[control.key] = var
                ttk.Label(self.controls_scroll.body, text=control.label).grid(row=row, column=0, sticky="w", pady=5)
                combo = ttk.Combobox(self.controls_scroll.body, textvariable=var, values=control.options, state="readonly")
                combo.grid(row=row, column=1, columnspan=2, sticky="ew", padx=8, pady=5)
                combo.bind("<<ComboboxSelected>>", lambda _event: self._on_parameter_changed())
            elif control.kind == "check":
                var = tk.BooleanVar(value=bool(control.default))
                self.param_vars[control.key] = var
                ttk.Checkbutton(self.controls_scroll.body, text=control.label, variable=var, command=self._on_parameter_changed).grid(row=row, column=0, columnspan=3, sticky="w", pady=5)
            if control.help_text:
                ttk.Label(self.controls_scroll.body, text=control.help_text, style="Muted.TLabel", wraplength=300).grid(row=row + 1, column=0, columnspan=3, sticky="ew")
        self.controls_scroll.body.grid_columnconfigure(1, weight=1)

    def _on_parameter_changed(self) -> None:
        self.preset_var.set("Manual")
        self.schedule_preview()

    def _params(self) -> dict[str, Any]:
        feature = self.feature_by_key[self.active_feature_key.get()]
        params: dict[str, Any] = {}
        for control in feature.controls:
            value = self.param_vars[control.key].get()
            if control.kind == "slider" and control.integer:
                params[control.key] = int(round(float(value)))
            elif control.kind == "slider":
                params[control.key] = float(value)
            else:
                params[control.key] = value
        return params

    def apply_preset(self) -> None:
        preset = self.preset_var.get()
        feature = self.feature_by_key[self.active_feature_key.get()]
        values = feature.presets.get(preset)
        if not values:
            self.schedule_preview()
            return
        for key, value in values.items():
            if key in self.param_vars:
                self.param_vars[key].set(value)
        self.schedule_preview()

    def reset_parameters(self) -> None:
        feature = self.feature_by_key[self.active_feature_key.get()]
        self.preset_var.set("Manual")
        for control in feature.controls:
            if control.key in self.param_vars:
                self.param_vars[control.key].set(control.default)
        self.schedule_preview()
        self._set_status("Parameter fitur aktif direset ke nilai awal.")

    # ------------------------------------------------------------------
    # Image management
    # ------------------------------------------------------------------
    def open_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="Pilih gambar",
            filetypes=(
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ),
        )
        if not filename:
            return
        try:
            with Image.open(filename) as pil:
                pil.load()
                image = np.asarray(pil.convert("RGB"), dtype=np.uint8).copy()
        except Exception as exc:  # pragma: no cover - UI error path
            messagebox.showerror("Gagal Membuka Gambar", str(exc))
            return
        self.current_path = Path(filename)
        self.original_image = image
        self.base_image = image.copy()
        self.processed_image = None
        pixel_count = int(image.shape[0] * image.shape[1])
        self.history.limit = LARGE_IMAGE_UNDO_LIMIT if pixel_count >= LARGE_IMAGE_UNDO_PIXEL_THRESHOLD else HISTORY_LIMIT
        self.history.clear()
        self._clear_crop_selection()
        self._update_info()
        self.refresh_canvases()
        if pixel_count >= LARGE_IMAGE_UNDO_PIXEL_THRESHOLD:
            self._set_status(
                f"Loaded full resolution: {self.current_path.name} ({image.shape[1]} x {image.shape[0]} px). "
                f"Undo dibatasi {self.history.limit} state agar gambar besar tetap lebih stabil."
            )
        else:
            self._set_status(f"Loaded full resolution: {self.current_path.name}")

    def save_image(self) -> None:
        image = self._current_result()
        if image is None:
            messagebox.showwarning("Tidak Ada Gambar", "Buka gambar terlebih dahulu.")
            return
        filename = filedialog.asksaveasfilename(
            title="Simpan hasil edit",
            defaultextension=".png",
            filetypes=(("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("BMP", "*.bmp"), ("TIFF", "*.tif")),
        )
        if not filename:
            return
        self._save_array_to_file(image, filename, self.save_quality_var.get())

    def save_with_quality(self) -> None:
        quality = simpledialog.askinteger("JPEG Quality", "Masukkan kualitas JPEG 1-100:", initialvalue=self.save_quality_var.get(), minvalue=1, maxvalue=100)
        if quality is None:
            return
        self.save_quality_var.set(quality)
        self.save_image()

    def _save_array_to_file(self, image: np.ndarray, filename: str, quality: int) -> None:
        try:
            pil = Image.fromarray(image)
            suffix = Path(filename).suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                pil.save(filename, quality=int(np.clip(quality, 1, 100)), optimize=True)
            else:
                pil.save(filename)
        except Exception as exc:  # pragma: no cover - UI error path
            messagebox.showerror("Gagal Menyimpan", str(exc))
            return
        self._set_status(f"Hasil disimpan: {Path(filename).name}")

    def reset_to_original(self) -> None:
        if self.original_image is None:
            return
        if self.base_image is not None:
            self.history.push(self.base_image)
        self.base_image = self.original_image.copy()
        self.processed_image = None
        self._clear_crop_selection()
        self._update_info()
        self.refresh_canvases()
        self._set_status("Gambar dikembalikan ke kondisi awal pada resolusi penuh.")

    def _current_result(self) -> Optional[np.ndarray]:
        if self.processed_image is not None:
            return self.processed_image
        return self.base_image

    # ------------------------------------------------------------------
    # Preview and apply
    # ------------------------------------------------------------------
    def schedule_preview(self) -> None:
        if self._update_job is not None:
            self.after_cancel(self._update_job)
        self._update_job = self.after(120, self.preview_active_feature if self.live_preview_var.get() else self.refresh_canvases)

    def preview_active_feature(self) -> None:
        self._update_job = None
        if self.base_image is None:
            self.refresh_canvases()
            return
        feature = self.feature_by_key[self.active_feature_key.get()]
        try:
            result = feature.processor(self.base_image, self._params())
            message = f"Preview full resolution: {feature.name}"
            if isinstance(result, tuple):
                image, message = result
            else:
                image = result
            self.processed_image = image
            self.refresh_canvases()
            self._set_status(message)
        except Exception as exc:
            self.processed_image = None
            self.refresh_canvases()
            self._set_status(f"Error pada {feature.name}: {exc}")

    def apply_active_feature(self) -> None:
        if self.base_image is None:
            messagebox.showwarning("Tidak Ada Gambar", "Buka gambar terlebih dahulu.")
            return
        self.preview_active_feature()
        if self.processed_image is None:
            return
        feature = self.feature_by_key[self.active_feature_key.get()]
        self.history.push(self.base_image)
        self.base_image = self.processed_image
        self.processed_image = None
        self._clear_crop_selection()
        self._update_info()
        self.refresh_canvases()
        self._set_status(f"Applied full resolution: {feature.name}")

    def quick_apply(self, feature_key: str, params: Optional[dict[str, Any]] = None) -> None:
        if feature_key not in self.feature_by_key:
            return
        self.feature_tree.selection_set(feature_key)
        self._select_feature(feature_key)
        if params:
            for key, value in params.items():
                if key in self.param_vars:
                    self.param_vars[key].set(value)
        self.apply_active_feature()

    def cancel_preview(self) -> None:
        if self.base_image is None:
            return
        self.processed_image = None
        self._clear_crop_selection()
        self.refresh_canvases()
        self._set_status("Preview dibatalkan. Gambar kembali ke state terakhir yang sudah di-apply.")

    def undo(self) -> None:
        if self.base_image is None:
            return
        restored = self.history.undo(self.base_image)
        if restored is None:
            self._set_status("Undo stack kosong.")
            return
        self.base_image = restored
        self.processed_image = None
        self._clear_crop_selection()
        self._update_info()
        self.refresh_canvases()
        self._set_status("Undo berhasil.")

    def redo(self) -> None:
        if self.base_image is None:
            return
        restored = self.history.redo(self.base_image)
        if restored is None:
            self._set_status("Redo stack kosong.")
            return
        self.base_image = restored
        self.processed_image = None
        self._clear_crop_selection()
        self._update_info()
        self.refresh_canvases()
        self._set_status("Redo berhasil.")

    # ------------------------------------------------------------------
    # Crop handling
    # ------------------------------------------------------------------
    def _start_crop_drag(self, event: tk.Event) -> None:
        if self._current_result() is None:
            return
        self._crop_start_canvas = (event.x, event.y)
        self._crop_end_canvas = (event.x, event.y)
        if self._crop_rect_id is not None:
            self.after_canvas.delete(self._crop_rect_id)
        self._crop_rect_id = self.after_canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#38bdf8", width=2, dash=(4, 2))

    def _drag_crop(self, event: tk.Event) -> None:
        if self._crop_start_canvas is None or self._crop_rect_id is None:
            return
        self._crop_end_canvas = (event.x, event.y)
        x1, y1 = self._crop_start_canvas
        self.after_canvas.coords(self._crop_rect_id, x1, y1, event.x, event.y)

    def _finish_crop_drag(self, event: tk.Event) -> None:
        if self._crop_start_canvas is None:
            return
        self._crop_end_canvas = (event.x, event.y)
        self._set_status("Area crop dipilih. Klik tombol 'Crop Area' untuk menerapkan.")

    def crop_selection(self) -> None:
        image = self._current_result()
        if image is None:
            return
        if self._crop_start_canvas is None or self._crop_end_canvas is None:
            messagebox.showinfo("Crop", "Drag area pada panel After terlebih dahulu.")
            return
        x1, y1 = self._canvas_to_image(self._crop_start_canvas)
        x2, y2 = self._canvas_to_image(self._crop_end_canvas)
        try:
            cropped = ip.crop(image, x1, y1, x2, y2)
        except Exception as exc:
            messagebox.showerror("Crop Gagal", str(exc))
            return
        if self.base_image is not None:
            self.history.push(self.base_image)
        self.base_image = cropped
        self.processed_image = None
        self._clear_crop_selection()
        self._update_info()
        self.refresh_canvases()
        self._set_status(f"Crop applied: {cropped.shape[1]} x {cropped.shape[0]} px")

    def _canvas_to_image(self, point: tuple[int, int]) -> tuple[int, int]:
        x, y = point
        ox, oy = self._after_offset
        scale = max(self._after_scale, 1e-9)
        return int(round((x - ox) / scale)), int(round((y - oy) / scale))

    def _clear_crop_selection(self) -> None:
        self._crop_start_canvas = None
        self._crop_end_canvas = None
        if hasattr(self, "after_canvas") and self._crop_rect_id is not None:
            try:
                self.after_canvas.delete(self._crop_rect_id)
            except tk.TclError:
                pass
        self._crop_rect_id = None

    # ------------------------------------------------------------------
    # Histogram and ML
    # ------------------------------------------------------------------
    def show_histogram(self) -> None:
        if self.original_image is None or self._current_result() is None:
            messagebox.showwarning("Histogram", "Buka gambar terlebih dahulu.")
            return
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        before = self.original_image if self.show_original_before_var.get() else self.base_image
        after = self._current_result()
        assert before is not None and after is not None
        before_hist = ip.compute_histograms(before)
        after_hist = ip.compute_histograms(after)

        win = tk.Toplevel(self)
        win.title("Histogram Before vs After")
        win.geometry("900x620")
        fig = Figure(figsize=(9, 6), dpi=100)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        xs = np.arange(256)
        for name in ["gray", "R", "G", "B"]:
            if name in before_hist:
                ax1.plot(xs, before_hist[name], label=name)
            if name in after_hist:
                ax2.plot(xs, after_hist[name], label=name)
        ax1.set_title("Before Histogram")
        ax2.set_title("After Histogram")
        ax1.set_xlim(0, 255)
        ax2.set_xlim(0, 255)
        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def recognize_object(self) -> None:
        image = self._current_result()
        if image is None:
            messagebox.showwarning("CNN", "Buka gambar terlebih dahulu.")
            return
        model_name = self.cnn_model_var.get() or DEFAULT_MODEL_NAME
        try:
            top_k = max(1, min(int(self.cnn_top_k_var.get()), 10))
        except Exception:
            top_k = 5
            self.cnn_top_k_var.set(top_k)
        try:
            from .ml import CNNRecognizer

            if model_name not in self._cnn_recognizers:
                self._set_status(f"Memuat model CNN {model_name}...")
                self.update_idletasks()
                self._cnn_recognizers[model_name] = CNNRecognizer(model_name=model_name)
            predictions = self._cnn_recognizers[model_name].predict(image, top_k=top_k)
        except Exception as exc:
            messagebox.showinfo(
                "CNN belum aktif",
                "Fitur CNN membutuhkan TensorFlow. Install dengan:\n\npip install -r requirements-ml.txt\n\n"
                f"Model yang dipilih: {model_name}\n\nDetail error:\n" + str(exc),
            )
            self._set_status("CNN belum aktif. Install requirements-ml.txt untuk memakai fitur ini.")
            return
        try:
            message = f"Model: {model_name}\n\n" + format_predictions(predictions)
        except Exception as exc:
            messagebox.showerror("CNN", "Gagal membaca format hasil prediksi CNN:\n" + str(exc))
            self._set_status("Format hasil CNN tidak didukung.")
            return
        messagebox.showinfo("CNN Object Recognition", message)
        self._set_status(f"CNN recognition selesai dengan {model_name}.")

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def refresh_canvases(self) -> None:
        if not hasattr(self, "before_canvas"):
            return
        if self.original_image is None:
            self._draw_empty_canvases()
            return
        before_image = self.original_image if self.show_original_before_var.get() else self.base_image
        after_image = self._current_result()
        if before_image is not None:
            self._draw_image(self.before_canvas, before_image, "before")
        if after_image is not None:
            self._draw_image(self.after_canvas, after_image, "after")

    def _draw_empty_canvases(self) -> None:
        if not hasattr(self, "before_canvas"):
            return
        for canvas, title in [(self.before_canvas, "Before"), (self.after_canvas, "After / Live Result")]:
            canvas.delete("all")
            w = max(canvas.winfo_width(), 300)
            h = max(canvas.winfo_height(), 260)
            canvas.create_text(w // 2, h // 2 - 12, text=title, fill="#e5e7eb", font=("Segoe UI", 16, "bold"))
            canvas.create_text(w // 2, h // 2 + 18, text="Load gambar JPG/PNG/BMP untuk mulai", fill="#94a3b8", font=("Segoe UI", 10))

    def _draw_image(self, canvas: tk.Canvas, image: np.ndarray, target: str) -> None:
        canvas.delete("all")
        w = max(canvas.winfo_width(), 1)
        h = max(canvas.winfo_height(), 1)
        preview, scale, ox, oy = ip.resize_to_fit(image, w, h)
        photo = ImageTk.PhotoImage(Image.fromarray(preview))
        canvas.create_image(ox, oy, image=photo, anchor="nw")
        canvas.create_rectangle(ox, oy, ox + preview.shape[1], oy + preview.shape[0], outline="#334155")
        canvas.create_text(12, 12, text=f"{image.shape[1]} x {image.shape[0]} px", fill="#e2e8f0", anchor="nw", font=("Segoe UI", 9, "bold"))
        if target == "before":
            self._before_photo = photo
            self._before_scale = scale
        else:
            self._after_photo = photo
            self._after_scale = scale
            self._after_offset = (ox, oy)
            if self._crop_start_canvas and self._crop_end_canvas:
                x1, y1 = self._crop_start_canvas
                x2, y2 = self._crop_end_canvas
                self._crop_rect_id = canvas.create_rectangle(x1, y1, x2, y2, outline="#38bdf8", width=2, dash=(4, 2))

    def _update_info(self) -> None:
        if self.base_image is None:
            self.info_var.set("Belum ada gambar.")
            return
        h, w = self.base_image.shape[:2]
        name = self.current_path.name if self.current_path else "Untitled"
        self.info_var.set(f"{name}  •  current: {w} x {h}px  •  undo: {'yes' if self.history.can_undo else 'no'}  •  redo: {'yes' if self.history.can_redo else 'no'}")

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)
        self._update_info()

    def show_help(self) -> None:
        messagebox.showinfo(
            "Panduan Singkat",
            "1. Load gambar.\n"
            "2. Pilih fitur di sidebar kiri.\n"
            "3. Atur parameter di panel kanan. Jika Live Preview aktif, hasil langsung muncul.\n"
            "4. Klik Apply untuk menyimpan efek ke state edit.\n"
            "5. Gunakan Undo/Redo, Histogram, Crop Area, dan Save sesuai kebutuhan.\n\n"
            "Tip: untuk crop, drag area pada panel After lalu klik Crop Area.\n"
            "Gambar besar dibuka dalam resolusi penuh; canvas hanya mengecilkan tampilan preview.",
        )


def run() -> None:
    app = MiniPhotoshopApp()
    app.mainloop()
