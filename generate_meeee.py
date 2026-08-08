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
import re
import math
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import xml.etree.ElementTree as ET

def remove_orange_background(img_rgb):
    """
    Removes the studio orange/yellow background cleanly in RGB space.
    Returns RGB array and foreground mask (1 for foreground, 0 for background).
    """
    arr = np.array(img_rgb, dtype=np.float32)
    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]
    
    # Orange/Yellow signature: strong Red (> 140), elevated Green (> 90), suppressed Blue (< 100), and large R-B separation
    is_orange = (R > 135) & (G > 85) & (B < 110) & ((R - B) > 50) & ((R - G) > 20) & ((R - G) < 120)
    
    # Convert binary mask to float and soften boundary slightly to prevent jagged hair edges
    bg_mask = is_orange.astype(np.float32)
    bg_img = Image.fromarray((bg_mask * 255).astype(np.uint8))
    # Fill small internal pinholes in background and blur slightly for smooth cutout
    bg_smooth = np.array(bg_img.filter(ImageFilter.GaussianBlur(radius=1.5)), dtype=np.float32) / 255.0
    
    fg_mask = 1.0 - np.clip(bg_smooth * 1.5, 0.0, 1.0)
    return arr, fg_mask

def enhance_portrait_features(arr_rgb, fg_mask, target_w=300, target_h=307):
    """
    Engineers the tonal harmony to match the crisp, high-contrast style of the original banner.
    - Lifts face highlights to clean solid whites
    - Preserves eyes and transparent eyeglass details
    - Modulates suit jacket texture into an architectural halftone grid
    - Adds crisp edge contours around hair, hand, and suit lapels
    """
    # Convert RGB to luminance, boosting green/blue slightly to penetrate yellow glasses tint and brighten eye area
    R = arr_rgb[:, :, 0] / 255.0
    G = arr_rgb[:, :, 1] / 255.0
    B = arr_rgb[:, :, 2] / 255.0
    
    # Custom luminance weighting that preserves eye sharpness behind yellow glasses
    lum = 0.25 * R + 0.55 * G + 0.20 * B
    
    # Extract sharp structural edges (lapel outlines, glasses frame, hair strands, fingers)
    lum_img = Image.fromarray((lum * 255).astype(np.uint8))
    blur_mid = np.array(lum_img.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32) / 255.0
    blur_fine = np.array(lum_img.filter(ImageFilter.GaussianBlur(radius=0.8)), dtype=np.float32) / 255.0
    
    edge_fine = lum - blur_fine
    edge_mid  = lum - blur_mid
    
    # Detect boundary outlines of foreground against black background
    fg_edge = np.array(Image.fromarray((fg_mask * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    outer_contour = (fg_edge > 0.2).astype(np.float32)
    
    # Enhance local facial contrast
    h, w = lum.shape
    y_idx, x_idx = np.indices((h, w))
    
    # Region definitions
    is_jacket = (y_idx > h * 0.55) & (lum < 0.25)
    is_face = (y_idx >= h * 0.15) & (y_idx <= h * 0.65) & (x_idx >= w * 0.18) & (x_idx <= w * 0.82)
    
    # Tone mapping: make jacket mid-tones structured instead of solid dead black
    enhanced = np.copy(lum)
    
    # Boost face highlights so cheeks and forehead render crisp and clean like original
    enhanced = np.where(is_face, enhanced * 1.25 + edge_fine * 2.2 + edge_mid * 1.2, enhanced)
    
    # Give jacket a sleek 30%-50% tone so it renders as a sharp halftone mesh with glowing lapell contours
    jacket_detail = np.clip(0.25 + (lum * 1.5) + (edge_mid * 3.0), 0.05, 0.60)
    enhanced = np.where(is_jacket, jacket_detail, enhanced)
    
    # Add crisp outer boundary highlight line around hair and shoulders
    enhanced = np.where(outer_contour > 0.5, np.maximum(enhanced, 0.85), enhanced)
    
    # Clamp and mask out background completely (zero dots in background!)
    enhanced = np.clip(enhanced, 0.0, 1.0) * (fg_mask > 0.3).astype(np.float32)
    return enhanced

def apply_8x8_bayer_halftone(intensity_map, is_dark_mode=True):
    """
    Applies an authentic 8x8 Ordered Bayer Dither Matrix, matching the exact digital glitch style
    of the original GitHub profile banner.
    """
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
    
    h, w = intensity_map.shape
    tiled_bayer = np.tile(bayer_8x8, (int(np.ceil(h / 8)), int(np.ceil(w / 8))))[:h, :w]
    
    binary = np.zeros((h, w), dtype=np.uint8)
    
    if is_dark_mode:
        # High intensity features become white/purple dots
        # Tone curve adjustment for optimal dot density
        adjusted = np.power(intensity_map, 0.9)
        binary[adjusted > (tiled_bayer * 0.85 + 0.05)] = 1
    else:
        # For light theme, dark features and outlines become purple dots on crisp white canvas
        inv = 1.0 - intensity_map
        # Ensure pure background remains zero dots (white canvas)
        inv[intensity_map == 0] = 0
        adjusted = np.power(inv, 0.85)
        binary[adjusted > (tiled_bayer * 0.82 + 0.06)] = 1
        
    # Enforce sharp 1px framing border box around canvas just like original banner
    binary[0:1, :] = 1; binary[-1:, :] = 1; binary[:, 0:1] = 1; binary[:, -1:] = 1
    return binary

def serialize_runs_to_path(runs):
    if not runs:
        return "M0 0h1v1h-1z"
    parts = [f"M{x} {y}h{l}v1h-{l}z" for x, y, l in runs]
    return "".join(parts)

def update_svg_file(source_svg, dest_svg, binary_canvas_300x307):
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
            
    with open(source_svg, 'r', encoding='utf-8') as f:
        content = f.read()
        
    tree = ET.ElementTree(ET.fromstring(content))
    root = tree.getroot()
    
    portrait_group = None
    for elem in root.iter():
        if elem.tag.endswith('g') and elem.attrib.get('transform') == "translate(50,86) scale(1.2400,1.4471)":
            portrait_group = elem
            break
            
    if portrait_group is None:
        print(f"ERROR: Portrait group not found in {source_svg}")
        return False
        
    set1, set2 = [], []
    for child in portrait_group:
        if child.tag.endswith('g'):
            if child.find('{*}animate') is not None or child.find('animate') is not None or child.attrib.get('opacity') == '0':
                set1.append(child)
            elif child.find('{*}animateTransform') is not None or child.find('animateTransform') is not None:
                set2.append(child)
                
    runs_sorted = sorted(runs, key=lambda r: (r[1], r[0]))
    total_runs = len(runs_sorted)
    
    if set1:
        chunk_1 = max(1, total_runs // len(set1))
        for i, grp in enumerate(set1):
            path = grp.find('{*}path')
            if path is None: path = grp.find('path')
            if path is not None:
                s = i * chunk_1
                e = (i + 1) * chunk_1 if i < len(set1) - 1 else total_runs
                path.set('d', serialize_runs_to_path(runs_sorted[s:e]))
                
    if set2:
        chunk_2 = max(1, total_runs // len(set2))
        for i, grp in enumerate(set2):
            path = grp.find('{*}path')
            if path is None: path = grp.find('path')
            if path is not None:
                s = i * chunk_2
                e = (i + 1) * chunk_2 if i < len(set2) - 1 else total_runs
                path.set('d', serialize_runs_to_path(runs_sorted[s:e]))
                
    tree.write(dest_svg, encoding="UTF-8", xml_declaration=True)
    print(f"Updated live SVG artwork: {dest_svg}")
    return True

if __name__ == "__main__":
    photo_path = r"c:\Users\M.Shahzaib\Desktop\updated github profile design\meeee.png"
    if not os.path.exists(photo_path):
        print(f"Error: Photo not found at {photo_path}")
        exit(1)
        
    print("Loading high-resolution studio photo meeee.png...")
    orig = Image.open(photo_path).convert('RGB')
    
    # Precision cropping and sizing to fit 300x307 canvas
    target_w, target_h = 300, 307
    img_w, img_h = orig.size
    aspect_target = target_w / target_h
    aspect_img = img_w / img_h
    if aspect_img > aspect_target:
        new_w = int(img_h * aspect_target)
        left = (img_w - new_w) // 2
        img = orig.crop((left, 0, left + new_w, img_h))
    else:
        new_h = int(img_w / aspect_target)
        top = int((img_h - new_h) * 0.08)  # keep hair crown well framed
        img = orig.crop((0, top, img_w, top + new_h))
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    print("Executing 100% color-based studio background segmentation...")
    arr_rgb, fg_mask = remove_orange_background(img)
    
    print("Engineering facial clarity, eyeglass contrast, and jacket halftone texture...")
    intensity = enhance_portrait_features(arr_rgb, fg_mask, target_w, target_h)
    
    print("Applying authentic 8x8 Bayer Ordered Halftoning...")
    binary_dark = apply_8x8_bayer_halftone(intensity, is_dark_mode=True)
    binary_light = apply_8x8_bayer_halftone(intensity, is_dark_mode=False)
    
    print(f"Dark portrait dot density: {np.mean(binary_dark):.2%}, Light portrait dot density: {np.mean(binary_light):.2%}")
    
    # Save verification media directly to desktop workspace AND artifacts directory
    artifacts_dir = r"C:/Users/M.Shahzaib/.gemini/antigravity/brain/d4cc5c92-ddf6-4f25-9f64-bcabfa9266d5"
    
    dark_preview = np.full((target_h, target_w, 3), [10, 16, 31], dtype=np.uint8)
    dark_preview[binary_dark == 1] = [167, 139, 250]
    p_dark_1 = r"c:\Users\M.Shahzaib\Desktop\updated github profile design\meeee_clear_dark.png"
    p_dark_2 = os.path.join(artifacts_dir, "meeee_clear_dark.png")
    Image.fromarray(dark_preview).save(p_dark_1)
    Image.fromarray(dark_preview).save(p_dark_2)
    
    light_preview = np.full((target_h, target_w, 3), [248, 250, 252], dtype=np.uint8)
    light_preview[binary_light == 1] = [124, 58, 237]
    p_light_1 = r"c:\Users\M.Shahzaib\Desktop\updated github profile design\meeee_clear_light.png"
    p_light_2 = os.path.join(artifacts_dir, "meeee_clear_light.png")
    Image.fromarray(light_preview).save(p_light_1)
    Image.fromarray(light_preview).save(p_light_2)
    print(f"Saved crystal-clear preview images:\n  -> {p_dark_1}\n  -> {p_light_1}")
    
    # Update live SVG files
    backup_dark = r"c:\Users\M.Shahzaib\Desktop\updated github profile design\arifhaxn-main\dark_original_backup.svg"
    live_dark = r"c:\Users\M.Shahzaib\Desktop\updated github profile design\arifhaxn-main\dark.svg"
    update_svg_file(backup_dark, live_dark, binary_dark)
    
    backup_light = r"c:\Users\M.Shahzaib\Desktop\updated github profile design\arifhaxn-main\light_original_backup.svg"
    live_light = r"c:\Users\M.Shahzaib\Desktop\updated github profile design\arifhaxn-main\light.svg"
    update_svg_file(backup_light, live_light, binary_light)
    print("\nSUCCESS! Both dark.svg and light.svg have been replaced with your crystal-clear portrait!")
