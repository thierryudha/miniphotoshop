
import cv2
import numpy as np
from mini_photoshop import image_processor as ip

def test_conversions():
    print("--- Testing Grayscale (H, W) ---")
    gray = np.zeros((100, 100), dtype=np.uint8)
    print(f"Input: {gray.shape}, ndim: {gray.ndim}")
    rgb = ip.gray_to_rgb(gray)
    print(f"Result gray_to_rgb: {rgb.shape}, ndim: {rgb.ndim}")
    
    print("\n--- Testing Grayscale (H, W, 1) ---")
    gray1 = np.zeros((100, 100, 1), dtype=np.uint8)
    print(f"Input: {gray1.shape}, ndim: {gray1.ndim}")
    rgb1 = ip.gray_to_rgb(gray1)
    print(f"Result gray_to_rgb: {rgb1.shape}, ndim: {rgb1.ndim} (BUG if not 3 channels)")
    
    print("\n--- Testing RGB (H, W, 3) ---")
    rgb_in = np.zeros((100, 100, 3), dtype=np.uint8)
    print(f"Input: {rgb_in.shape}, ndim: {rgb_in.ndim}")
    gray_out = ip.to_gray(rgb_in)
    print(f"Result to_gray: {gray_out.shape}, ndim: {gray_out.ndim}")
    
    print("\n--- Testing RGBA (H, W, 4) ---")
    rgba_in = np.zeros((100, 100, 4), dtype=np.uint8)
    print(f"Input: {rgba_in.shape}, ndim: {rgba_in.ndim}")
    try:
        gray_out = ip.to_gray(rgba_in)
        print(f"Result to_gray: {gray_out.shape}, ndim: {gray_out.ndim}")
    except Exception as e:
        print(f"Result to_gray: FAILED with {e}")

    print("\n--- Testing rgb_to_grayscale on RGB ---")
    rgb_in = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    res = ip.rgb_to_grayscale(rgb_in)
    print(f"Input: {rgb_in.shape}, Result: {res.shape}")
    is_gray = np.all(res[:,:,0] == res[:,:,1]) and np.all(res[:,:,1] == res[:,:,2])
    print(f"Is actually grayscale (R=G=B): {is_gray}")

if __name__ == "__main__":
    test_conversions()
