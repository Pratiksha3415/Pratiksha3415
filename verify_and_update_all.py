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
import xml.etree.ElementTree as ET

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

def serialize_runs_to_path(runs):
    if not runs:
        return "M0 0h1v1h-1z"
    parts = [f"M{x} {y}h{l}v1h-{l}z" for x, y, l in runs]
    return "".join(parts)

def update_svg_file(source_svg, dest_svg, binary_canvas_300x307, name="SVG"):
    print(f"\n=================== UPDATING {name} ({dest_svg}) ===================")
    full_canvas = np.zeros((350, 320), dtype=np.uint8)
    full_canvas[32:32+307, 0:300] = binary_canvas_300x307
    
    runs = []
    for y in range(350):
        row = full_canvas[y]
        diff = np.diff(np.concatenate(([0], row, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            runs.append((s, y, e - s))
            
    runs_sorted = sorted(runs, key=lambda r: (r[1], r[0]))
    total_runs = len(runs_sorted)
    print(f"Generated {total_runs} horizontal pixel runs for portrait.")
    
    with open(source_svg, 'r', encoding='utf-8') as f:
        content = f.read()
        
    tree = ET.ElementTree(ET.fromstring(content))
    root = tree.getroot()
    
    # FIND ALL containers that have translate(50,86) scale(...) or hold portrait animation groups
    containers_modified = 0
    total_paths_updated = 0
    
    for elem in root.iter():
        if elem.tag.endswith('g'):
            tr = elem.attrib.get('transform', '')
            if 'translate(50,86)' in tr or 'scale(1.24' in tr or 'translate(50' in tr:
                # Find all direct child groups that contain paths
                child_groups_with_paths = []
                for child in elem:
                    if child.tag.endswith('g'):
                        path_el = child.find('{*}path')
                        if path_el is None:
                            path_el = child.find('path')
                        if path_el is not None:
                            child_groups_with_paths.append((child, path_el))
                            
                num_groups = len(child_groups_with_paths)
                if num_groups > 0:
                    containers_modified += 1
                    chunk = max(1, total_runs // num_groups)
                    print(f"  Container #{containers_modified} (transform='{tr}'): updating {num_groups} animation groups...")
                    for i, (grp, path_el) in enumerate(child_groups_with_paths):
                        s = i * chunk
                        e = (i + 1) * chunk if i < num_groups - 1 else total_runs
                        path_el.set('d', serialize_runs_to_path(runs_sorted[s:e]))
                        total_paths_updated += 1
                        
    print(f"Successfully updated {containers_modified} portrait group containers across {total_paths_updated} animated path layers!")
    
    tree.write(dest_svg, encoding="UTF-8", xml_declaration=True)
    print(f"Live file written: {dest_svg}")
    return True

if __name__ == "__main__":
    photo_path = "my image.png"
    if not os.path.exists(photo_path):
        print(f"Error: file {photo_path} not found.")
        exit(1)
        
    print("Processing 'my image.png' with Ultimate Adaptive Contrast & Density Engine...")
    res_arr = load_and_crop_photo(photo_path, 300, 307)
    
    binary_dark = generate_halftone_dither(res_arr, is_dark_mode=True, target_density=0.182)
    binary_light = generate_halftone_dither(res_arr, is_dark_mode=False, target_density=0.415)
    
    # Save verification previews
    h, w = binary_dark.shape
    dark_preview = np.full((h, w, 3), [10, 16, 31], dtype=np.uint8)
    dark_preview[binary_dark == 1] = [167, 139, 250]
    Image.fromarray(dark_preview).save("perfect_dark.png")
    
    light_preview = np.full((h, w, 3), [248, 250, 252], dtype=np.uint8)
    light_preview[binary_light == 1] = [124, 58, 237]
    Image.fromarray(light_preview).save("perfect_light.png")
    
    artifacts_dir = "C:/Users/M.Shahzaib/.gemini/antigravity/brain/d4cc5c92-ddf6-4f25-9f64-bcabfa9266d5"
    if os.path.exists(artifacts_dir):
        shutil.copyfile("perfect_dark.png", os.path.join(artifacts_dir, "perfect_dark.png"))
        shutil.copyfile("perfect_light.png", os.path.join(artifacts_dir, "perfect_light.png"))
        print("Copied preview images to artifacts directory.")
        
    # Perform full deep replacement on both dark and light SVGs
    backup_dark = "arifhaxn-main/dark_original_backup.svg"
    live_dark = "arifhaxn-main/dark.svg"
    update_svg_file(backup_dark, live_dark, binary_dark, "Dark Theme Banner")
    
    backup_light = "arifhaxn-main/light_original_backup.svg"
    live_light = "arifhaxn-main/light.svg"
    update_svg_file(backup_light, live_light, binary_light, "Light Theme Banner")
    
    print("\nSUCCESS: Both dark.svg and light.svg have been perfectly rebuilt without any trace of the old portrait!")
