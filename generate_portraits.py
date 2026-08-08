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
import numpy as np
from PIL import Image, ImageFilter
import xml.etree.ElementTree as ET

# Generate 8x8 Bayer matrix for museum-quality digital halftoning
def get_bayer_matrix_8x8():
    b2 = np.array([[0, 2], [3, 1]])
    b4 = np.block([[4*b2 + 0, 4*b2 + 2], [4*b2 + 3, 4*b2 + 1]])
    b8 = np.block([[4*b4 + 0, 4*b4 + 2], [4*b4 + 3, 4*b4 + 1]])
    return (b8 + 0.5) / 64.0

def generate_master_portrait_arrays(photo_path):
    orig = Image.open(photo_path).convert('L')
    target_w, target_h = 300, 307  # Exact grid height matching original SVG layout
    
    img_w, img_h = orig.size
    aspect_target = target_w / target_h
    aspect_img = img_w / img_h
    
    if aspect_img > aspect_target:
        new_w = int(img_h * aspect_target)
        left = (img_w - new_w) // 2
        img = orig.crop((left, 0, left + new_w, img_h))
    else:
        new_h = int(img_w / aspect_target)
        top = int((img_h - new_h) * 0.15) # favor top/face area
        img = orig.crop((0, top, img_w, top + new_h))
        
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    
    # 1. EDGE DETECTION & TEXTURE MAPPING
    edges = np.array(img.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    blur_edges = np.array(Image.fromarray((edges*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=4)), dtype=np.float32) / 255.0
    
    # 2. STUDIO WALL BACKGROUND SUPPRESSION
    # Separate the figure (face, hand, hair, jacket) from the background studio wall
    y_idx, x_idx = np.indices((target_h, target_w))
    center_dist_x = np.abs(x_idx - target_w/2.0) / (target_w / 2.0)
    center_dist_y = y_idx / float(target_h)
    
    # Background wall signature: top half of canvas, low edge texture, relatively high brightness
    is_bg = (blur_edges < 0.035) & (arr > 0.35) & ((center_dist_x > 0.35) | (center_dist_y < 0.25))
    is_bg = is_bg | ((center_dist_y < 0.55) & (center_dist_x > 0.45) & (blur_edges < 0.05))
    
    fg_mask = 1.0 - is_bg.astype(np.float32)
    fg_mask_img = Image.fromarray((fg_mask*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3))
    fg_mask = np.array(fg_mask_img, dtype=np.float32) / 255.0

    # 3. ADAPTIVE LOCAL CONTRAST NORMALIZATION (Unsharp masking & detail boost)
    blur_mid = np.array(img.filter(ImageFilter.GaussianBlur(radius=7)), dtype=np.float32) / 255.0
    sharp_detail = (arr - blur_mid) * 1.55
    enhanced = np.clip(arr + sharp_detail, 0.0, 1.0)
    
    # 4. ORDERED HALFTONING VIA 8x8 BAYER MATRIX
    bayer = get_bayer_matrix_8x8()
    bayer_tiled = np.tile(bayer, (int(np.ceil(target_h/8)), int(np.ceil(target_w/8))))[:target_h, :target_w]
    
    # --- DARK MODE PORTRAIT ---
    signal_dark = enhanced * fg_mask
    signal_dark = np.clip(signal_dark * 1.18, 0.0, 1.0) # check facial highlights
    
    # Modulate jacket tones so clothing renders as architectural sparse texture rather than solid blobs
    jacket_mask = (y_idx > target_h * 0.65).astype(np.float32)
    signal_dark = np.where((jacket_mask > 0) & (edges > 0.06) & (fg_mask > 0.5), np.maximum(signal_dark, 0.55), signal_dark)
    
    thresh_dark = bayer_tiled * 0.75 + 0.12
    binary_dark = (signal_dark > thresh_dark).astype(np.uint8)
    binary_dark[fg_mask < 0.2] = 0
    
    # Add signature crisp 1px rectangular box frame around artwork
    binary_dark[0:1, :] = 1; binary_dark[-1:, :] = 1; binary_dark[:, 0:1] = 1; binary_dark[:, -1:] = 1
    
    # --- LIGHT MODE PORTRAIT ---
    inv_enhanced = 1.0 - enhanced
    signal_light = inv_enhanced * fg_mask
    signal_light = np.clip(signal_light * 1.22, 0.0, 1.0)
    
    thresh_light = bayer_tiled * 0.68 + 0.15
    binary_light = (signal_light > thresh_light).astype(np.uint8)
    binary_light[fg_mask < 0.2] = 0
    
    binary_light[0:1, :] = 1; binary_light[-1:, :] = 1; binary_light[:, 0:1] = 1; binary_light[:, -1:] = 1
    
    # Place on exact 350x320 SVG canvas grid at row offset 32 (matching original spatial translation)
    canvas_dark = np.zeros((350, 320), dtype=np.uint8)
    canvas_dark[32:32+target_h, 0:target_w] = binary_dark
    
    canvas_light = np.zeros((350, 320), dtype=np.uint8)
    canvas_light[32:32+target_h, 0:target_w] = binary_light
    
    return canvas_dark, canvas_light

def extract_horizontal_runs(canvas):
    runs = []
    for y in range(350):
        row = canvas[y]
        diff = np.diff(np.concatenate(([0], row, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            runs.append((s, y, e - s))
    return runs

def serialize_runs_to_path(runs):
    if not runs:
        return "M 0,0"
    parts = []
    for x, y, length in runs:
        parts.append(f"M {x},{y} h {length}")
    return " ".join(parts)

def update_svg_file(svg_path, new_canvas):
    print(f"Updating portrait section in {svg_path}...")
    runs = extract_horizontal_runs(new_canvas)
    print(f"  Extracted {len(runs)} horizontal runs (Active pixels: {np.sum(new_canvas)}).")
    
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Find portrait master group
    portrait_group = None
    for elem in root.iter():
        if elem.tag.endswith('g') and elem.attrib.get('transform') == "translate(50,86) scale(1.2400,1.4471)":
            portrait_group = elem
            break
            
    if portrait_group is None:
        print(f"ERROR: Could not find portrait group in {svg_path}")
        return False
        
    set1 = []
    set2 = []
    for child in portrait_group:
        if child.tag.endswith('g'):
            if child.find('{*}animate') is not None or child.find('animate') is not None or child.attrib.get('opacity') == '0':
                set1.append(child)
            elif child.find('{*}animateTransform') is not None or child.find('animateTransform') is not None:
                set2.append(child)
                
    print(f"  Found {len(set1)} Set 1 (fade-in) groups and {len(set2)} Set 2 (glitch animation) groups.")
    
    # Sort runs spatially by Y coordinate to feed smoothly into animation tiers
    runs_sorted = sorted(runs, key=lambda r: (r[1], r[0]))
    total_runs = len(runs_sorted)
    
    # --- POPULATE SET 1 (Fade-in materialization effect) ---
    if len(set1) > 0:
        chunk_size_1 = max(1, total_runs // len(set1))
        for i, grp in enumerate(set1):
            path = grp.find('{*}path')
            if path is None: path = grp.find('path')
            if path is not None:
                start = i * chunk_size_1
                end = (i + 1) * chunk_size_1 if i < len(set1) - 1 else total_runs
                path.set('d', serialize_runs_to_path(runs_sorted[start:end]))
                
    # --- POPULATE SET 2 (Continuous cyber glitch animation tiles) ---
    if len(set2) > 0:
        # Sort runs into vertical slices matching Set 2 glitch bands
        chunk_size_2 = max(1, total_runs // len(set2))
        for i, grp in enumerate(set2):
            path = grp.find('{*}path')
            if path is None: path = grp.find('path')
            if path is not None:
                start = i * chunk_size_2
                end = (i + 1) * chunk_size_2 if i < len(set2) - 1 else total_runs
                path.set('d', serialize_runs_to_path(runs_sorted[start:end]))
                
    # Save back to SVG retaining exact structure and animations
    tree.write(svg_path, encoding="UTF-8", xml_declaration=True)
    print(f"Successfully engineered and applied portrait to {svg_path}!")
    return True

if __name__ == "__main__":
    photo_file = r"C:/Users/M.Shahzaib/.gemini/antigravity/brain/d4cc5c92-ddf6-4f25-9f64-bcabfa9266d5/media__1785335440634.jpg"
    print("Executing Master Bayer-Adaptive Halftoning Pipeline...")
    dark_canvas, light_canvas = generate_master_portrait_arrays(photo_file)
    
    # Save high-resolution visual previews to verify quality
    dark_preview = np.full((350, 320, 3), [10, 16, 31], dtype=np.uint8) # #0A101F dark theme background
    dark_preview[dark_canvas == 1] = [167, 139, 250] # #A78BFA purple vibrant dots
    Image.fromarray(dark_preview).save(r"C:/Users/M.Shahzaib/.gemini/antigravity/brain/d4cc5c92-ddf6-4f25-9f64-bcabfa9266d5/master_preview_dark.png")
    
    light_preview = np.full((350, 320, 3), [248, 250, 252], dtype=np.uint8) # #F8FAFC light theme background
    light_preview[light_canvas == 1] = [124, 58, 237] # #7C3AED deep purple dots
    Image.fromarray(light_preview).save(r"C:/Users/M.Shahzaib/.gemini/antigravity/brain/d4cc5c92-ddf6-4f25-9f64-bcabfa9266d5/master_preview_light.png")
    print("Saved master_preview_dark.png and master_preview_light.png to artifacts.")
    
    # Apply to SVG files
    dark_svg = r"c:\Users\M.Shahzaib\Desktop\updated github profile design\arifhaxn-main\dark.svg"
    light_svg = r"c:\Users\M.Shahzaib\Desktop\updated github profile design\arifhaxn-main\light.svg"
    
    update_svg_file(dark_svg, dark_canvas)
    update_svg_file(light_svg, light_canvas)
    print("ALL DONE! Both dark.svg and light.svg have been perfectly re-engineered!")
