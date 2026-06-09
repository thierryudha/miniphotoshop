import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mini_photoshop import image_processor as ip

img = np.zeros((64, 64, 3), dtype=np.uint8)
img[16:48, 16:48] = [200, 120, 40]

settings = ip.LiveSettings(brightness=10, contrast=20, rotate=15, scale=1.0, translate_x=2, translate_y=-2)
out = ip.apply_live_settings(img, settings)
assert out.shape == img.shape
assert out.dtype == np.uint8

for fn in [
    ip.equalize_histogram,
    ip.sharpen,
    lambda x: ip.average_smoothing(x, 3),
    lambda x: ip.gaussian_blur(x, 3),
    lambda x: ip.median_filter(x, 3),
    ip.rgb_to_grayscale,
    lambda x: ip.threshold_binary(x, 127),
    lambda x: ip.edge_detection(x, 'Canny'),
    lambda x: ip.edge_detection(x, 'Sobel'),
    lambda x: ip.edge_detection(x, 'Prewitt'),
    lambda x: ip.edge_detection(x, 'Robert'),
    lambda x: ip.edge_detection(x, 'Laplacian'),
    lambda x: ip.edge_detection(x, 'Laplacian of Gaussian'),
    lambda x: ip.morphology(x, 'erosion', 3),
    lambda x: ip.morphology(x, 'dilation', 3),
    lambda x: ip.threshold_segmentation(x, 127),
    ip.edge_based_segmentation,
    lambda x: ip.region_based_segmentation(x, 3),
    lambda x: ip.simulate_jpeg(x, 70),
    lambda x: ip.quantize_colors(x, 4),
]:
    result = fn(img)
    assert result.ndim == 3
    assert result.dtype == np.uint8
    assert result.shape[2] == 3

cropped = ip.crop(img, 10, 10, 30, 30)
assert cropped.shape[:2] == (20, 20)
resized = ip.resize_image(img, 32, 16)
assert resized.shape[:2] == (16, 32)
ratio = ip.rle_compression_ratio(img)
assert ratio > 0
hist = ip.compute_histograms(img)
assert hist['R'].shape == (256,)
assert 'gray' not in hist
assert ip.select_histogram_channels(hist, 'all') == ('R', 'G', 'B')
assert ip.select_histogram_channels(hist, 'R') == ('R',)
gray_img = np.zeros((16, 16), dtype=np.uint8)
gray_hist = ip.compute_histograms(gray_img)
assert tuple(gray_hist) == ('gray',)
assert ip.select_histogram_channels(gray_hist, 'all') == ('gray',)
assert ip.select_histogram_channels(gray_hist, 'R') == ()
print('image_processor smoke tests passed')
