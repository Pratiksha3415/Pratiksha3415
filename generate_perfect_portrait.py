"""
DEPRECATED: superseded by rebuild_perfect_svgs_with_morph.py.

This script is an earlier draft of the portrait pipeline. It still contains
hardcoded Windows paths from the original author's machine and, in some
cases, points at the OLD source photo (meeee.png, now in
_old_source_backup/). Do not run this to regenerate the live portrait --
use:

    python rebuild_perfect_svgs_with_morph.py

Kept only for reference to the halftone/dither logic it was built from.
"""
import os
import math
import shutil
import numpy as np
from PIL import Image, ImageFilter, ImageOps

def load_and_crop_photo(photo_path, target_w=300, target_h=307):
    img = Image.open(photo_path)
    arr = np.array(img.convert("RGBA"), dtype=np.float32)
    
    alpha = arr[:, :, 3] / 255.0
    row_alpha = alpha.sum(axis=1)
    top_y = np.where(row_alpha > target_w * 0.1)[0][0]
    
    start_y = max(0, top_y - 25)
    orig_h, orig_w = arr.shape[:2]
    target_aspect = target_w / target_h
    desired_h = int(round(orig_w / target_aspect))
    
    if start_y + desired_h <= orig_h:
        cropped = img.crop((0, start_y, orig_w, start_y + desired_h))
    else:
        avail_h = orig_h - start_y
        desired_w = int(round(avail_h * target_aspect))
        left_x = (orig_w - desired_w) // 2
        cropped = img.crop((left_x, start_y, left_x + desired_w, orig_h))
        
    resized = cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)
    res_arr = np.array(resized, dtype=np.float32)
    return res_arr

def compute_adaptive_contrast_map(res_arr):
    rgb = res_arr[:, :, :3] / 255.0
    alpha = res_arr[:, :, 3] / 255.0
    
    alpha_img = Image.fromarray((alpha * 255).astype(np.uint8))
    alpha_smooth = np.array(alpha_img.filter(ImageFilter.GaussianBlur(radius=0.5)), dtype=np.float32) / 255.0
    fg_mask = (alpha_smooth > 0.15).astype(np.float32)
    
    lum = 0.28 * rgb[:, :, 0] + 0.57 * rgb[:, :, 1] + 0.15 * rgb[:, :, 2]
    
    lum_img = Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8))
    blur_large = np.array(lum_img.filter(ImageFilter.GaussianBlur(radius=10)), dtype=np.float32) / 255.0
    blur_med   = np.array(lum_img.filter(ImageFilter.GaussianBlur(radius=3)), dtype=np.float32) / 255.0
    blur_fine  = np.array(lum_img.filter(ImageFilter.GaussianBlur(radius=0.8)), dtype=np.float32) / 255.0
    
    detail_fine = lum - blur_fine
    detail_med  = lum - blur_med
    local_norm  = lum / (blur_large + 0.05)
    
    edge_detector = np.array(lum_img.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    edges = np.clip(edge_detector * 2.0, 0, 1) * fg_mask
    
    h, w = lum.shape
    y_idx, x_idx = np.indices((h, w))
    
    is_face = (y_idx >= h * 0.15) & (y_idx <= h * 0.65) & (x_idx >= w * 0.2) & (x_idx <= w * 0.8) & (fg_mask > 0.5)
    is_jacket = (y_idx > h * 0.60) & (fg_mask > 0.5)
    is_beard = (y_idx >= h * 0.45) & (y_idx <= h * 0.65) & (x_idx >= w * 0.3) & (x_idx <= w * 0.7) & (fg_mask > 0.5)
    
    return lum, local_norm, detail_fine, detail_med, edges, fg_mask

def generate_halftone_dither(res_arr, is_dark_mode=True, target_density=0.18):
    lum, local_norm, detail_fine, detail_med, edges, fg_mask = compute_adaptive_contrast_map(res_arr)
    
    bayer_8x8 = np.array([
        [ 0, 32,  8, 40,  2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44,  4, 36, 14, 46,  6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [ 3, 35, 11, 43,  1, 33,  9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47,  7, 39, 13, 45,  5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21]
    ], dtype=np.float32) / 64.0
    
    h, w = lum.shape
    tiled_bayer = np.tile(bayer_8x8, (int(np.ceil(h / 8)), int(np.ceil(w / 8))))[:h, :w]
    
    if is_dark_mode:
        base_tone = lum * 0.65 + (local_norm - 1.0) * 0.15 + detail_med * 1.8 + detail_fine * 1.4
        base_tone += edges * 0.35
        base_tone = np.clip(base_tone, 0, 1) * fg_mask
        
        low, high = -1.0, 1.0
        best_binary = None
        for _ in range(25):
            mid = (low + high) / 2.0
            thresh = (tiled_bayer * 0.75 + 0.12) + mid
            binary = (base_tone > thresh).astype(np.uint8) * (fg_mask > 0.3).astype(np.uint8)
            binary = np.where(edges > 0.45, 1, binary) * (fg_mask > 0.3).astype(np.uint8)
            binary[0, :] = 1; binary[-1, :] = 1; binary[:, 0] = 1; binary[:, -1] = 1
            
            density = binary.mean()
            if density < target_density:
                high = mid
            else:
                low = mid
        best_binary = binary
        
    else:
        inv_lum = 1.0 - lum
        base_tone = inv_lum * 0.75 + (1.0 - local_norm) * 0.15 - detail_med * 1.8 - detail_fine * 1.4
        base_tone += edges * 0.45
        base_tone = np.clip(base_tone, 0, 1) * fg_mask
        
        low, high = -1.0, 1.0
        best_binary = None
        for _ in range(25):
            mid = (low + high) / 2.0
            thresh = (tiled_bayer * 0.70 + 0.10) + mid
            binary = (base_tone > thresh).astype(np.uint8) * (fg_mask > 0.3).astype(np.uint8)
            binary = np.where(edges > 0.40, 1, binary) * (fg_mask > 0.3).astype(np.uint8)
            binary[0, :] = 1; binary[-1, :] = 1; binary[:, 0] = 1; binary[:, -1] = 1
            
            density = binary.mean()
            if density > target_density:
                low = mid
            else:
                high = mid
        best_binary = binary
        
    return best_binary

def print_statistics(name, binary):
    total = binary.size
    active = binary.sum()
    print(f"\n--- Statistical Quality Check: {name} ---")
    print(f"Total active dots: {active} out of {total} ({active/total:.2%})")
    
    runs = []
    h, w = binary.shape
    for y in range(h):
        row = binary[y]
        diff = np.diff(np.concatenate(([0], row, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            runs.append(e - s)
            
    runs = np.array(runs)
    if len(runs) > 0:
        single_pixels = (runs == 1).sum() / len(runs)
        print(f"Run count: {len(runs)}, Mean run length: {runs.mean():.2f}, Single-pixel runs: {single_pixels:.2%}")
    else:
        print("No runs found.")

if __name__ == "__main__":
    photo_path = "my image.png"
    if not os.path.exists(photo_path):
        print(f"Error: file {photo_path} not found.")
        exit(1)
        
    print("Processing photo with Ultimate Adaptive Contrast & Density Engine...")
    res_arr = load_and_crop_photo(photo_path, 300, 307)
    
    binary_dark = generate_halftone_dither(res_arr, is_dark_mode=True, target_density=0.182)
    binary_light = generate_halftone_dither(res_arr, is_dark_mode=False, target_density=0.415)
    
    print_statistics("Dark Theme Portrait", binary_dark)
    print_statistics("Light Theme Portrait", binary_light)
    
    h, w = binary_dark.shape
    dark_preview = np.full((h, w, 3), [10, 16, 31], dtype=np.uint8)
    dark_preview[binary_dark == 1] = [167, 139, 250]
    Image.fromarray(dark_preview).save("perfect_dark.png")
    
    light_preview = np.full((h, w, 3), [248, 250, 252], dtype=np.uint8)
    light_preview[binary_light == 1] = [124, 58, 237]
    Image.fromarray(light_preview).save("perfect_light.png")
    
    # Save to artifacts directory as well
    artifacts_dir = "C:/Users/M.Shahzaib/.gemini/antigravity/brain/d4cc5c92-ddf6-4f25-9f64-bcabfa9266d5"
    if os.path.exists(artifacts_dir):
        shutil.copyfile("perfect_dark.png", os.path.join(artifacts_dir, "perfect_dark.png"))
        shutil.copyfile("perfect_light.png", os.path.join(artifacts_dir, "perfect_light.png"))
        print(f"\nCopied preview images to artifacts directory: {artifacts_dir}")
        
    print("\nSaved high-resolution verification previews:\n  -> perfect_dark.png\n  -> perfect_light.png")
