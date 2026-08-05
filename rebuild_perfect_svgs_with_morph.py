import os
import shutil
import re
import math
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageFilter, ImageDraw

def strip_ns(tag):
    return tag.split('}')[-1] if '}' in tag else tag

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

def generate_halftone_dither(res_arr, is_dark_mode=True, target_density=0.182):
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
            
            density = binary.mean()
            if density < target_density:
                high = mid
            else:
                low = mid
        best_binary = binary
    else:
        inv_lum = 1.0 - lum
        base_tone = inv_lum * 0.80 + (1.0 - local_norm) * 0.15 - detail_med * 1.5 - detail_fine * 1.5
        base_tone += edges * 0.50
        base_tone = np.clip(base_tone, 0, 1) * fg_mask
        
        low, high = -1.0, 1.0
        best_binary = None
        for _ in range(25):
            mid = (low + high) / 2.0
            thresh = (tiled_bayer * 0.70 + 0.10) + mid
            binary = (base_tone > thresh).astype(np.uint8) * (fg_mask > 0.3).astype(np.uint8)
            binary = np.where(edges > 0.42, 1, binary) * (fg_mask > 0.3).astype(np.uint8)
            
            density = binary.mean()
            if density < target_density:
                high = mid
            else:
                low = mid
        best_binary = binary
        
    return best_binary

def binary_to_runs_with_coords(binary_300x307):
    # Embed inside 350x320 canvas with top-left offset (y=32, x=0) for perfect alignment with original layout
    full_canvas = np.zeros((350, 320), dtype=np.uint8)
    full_canvas[32:32+307, 0:300] = binary_300x307
    
    runs = []
    for y in range(350):
        row = full_canvas[y]
        diff = np.diff(np.concatenate(([0], row, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            length = e - s
            mid_x = s + length / 2.0
            runs.append((s, y, length, mid_x))
    return runs

def parse_runs_centroids(d_str):
    res = []
    for m in re.finditer(r'M(\d+)\s*(\d+)h(\d+)', d_str or ''):
        res.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return res

def serialize_runs(runs_list):
    if not runs_list:
        return "M0 0h1v1h-1z"
    # CRITICAL FIX: v1h-{l}z is REQUIRED for filled path rendering without a stroke attribute!
    parts = [f"M{x} {y}h{l}v1h-{l}z" for x, y, l, _ in runs_list]
    return "".join(parts)

def generate_z_coordinates(count=900):
    img = Image.new('L', (320, 350), 0)
    draw = ImageDraw.Draw(img)
    # Futuristic bold Z polygon centered at (150, 175), matching dimensions of previous triangle
    z_polygon = [
        (65, 90),
        (235, 90),
        (235, 124),
        (107, 226),
        (235, 226),
        (235, 260),
        (65, 260),
        (65, 226),
        (193, 124),
        (65, 124)
    ]
    draw.polygon(z_polygon, fill=255)
    mask = np.array(img)
    y_idx, x_idx = np.where(mask > 0)
    all_points = list(zip(x_idx, y_idx))
    
    best_pts = []
    best_diff = 999999
    for step in np.linspace(4.0, 4.6, 300):
        pts = set()
        y_vals = np.arange(90, 261, step)
        x_vals = np.arange(65, 236, step)
        for y in y_vals:
            for x in x_vals:
                ix, iy = int(round(x)), int(round(y))
                if 0 <= iy < 350 and 0 <= ix < 320 and mask[iy, ix] > 0:
                    pts.add((ix, iy))
        diff = abs(len(pts) - count)
        if diff < best_diff:
            best_diff = diff
            best_pts = sorted(list(pts), key=lambda p: (p[1], p[0]))
            if diff == 0:
                break
                
    if len(best_pts) > count:
        idx_to_keep = np.linspace(0, len(best_pts) - 1, count).astype(int)
        best_pts = [best_pts[i] for i in idx_to_keep]
    elif len(best_pts) < count:
        all_set = set(all_points) - set(best_pts)
        best_pts.extend(list(all_set)[:count - len(best_pts)])
        best_pts = sorted(best_pts, key=lambda p: (p[1], p[0]))
        
    return best_pts

def match_points_1to1(source_pts, target_pts):
    src = np.array(source_pts, dtype=np.float32)
    tgt = np.array(target_pts, dtype=np.float32)
    N = len(src)
    M = len(tgt)
    assigned_target = [None] * N
    available_indices = set(range(M))
    
    dists = np.sum((src[:, None, :] - tgt[None, :, :])**2, axis=-1)
    for i in range(N):
        avail_list = list(available_indices)
        if not avail_list:
            assigned_target[i] = tgt[0]
        else:
            sub_dists = dists[i, avail_list]
            best_idx = np.argmin(sub_dists)
            best_j = avail_list[best_idx]
            available_indices.remove(best_j)
            assigned_target[i] = tgt[best_j]
    return assigned_target

def process_and_rebuild_svg(backup_path, output_path, new_runs, is_dark, z_coords):
    print(f"\n==================== Rebuilding {output_path} ====================")
    with open(backup_path, 'r', encoding='utf-8') as f:
        content = f.read()
    tree = ET.ElementTree(ET.fromstring(content))
    root = tree.getroot()
    
    c1, c2, c3 = None, None, None
    for elem in root.iter():
        tag = strip_ns(elem.tag)
        # Ensure the VISUAL.MAP frame interior background and glowing border remain identical to dark mode in ALL themes!
        if tag == 'rect' and elem.attrib.get('width') == '400' and elem.attrib.get('height') == '492':
            if elem.attrib.get('fill') == 'none':
                elem.attrib['stroke'] = '#22D3EE'
            else:
                elem.attrib['fill'] = '#0A101F'
                elem.attrib['stroke'] = 'rgba(34,211,238,0.35)'
                
        if tag == 'g':
            tr = elem.attrib.get('transform', '')
            op = elem.attrib.get('opacity', '1')
            if 'translate(50,86)' in tr or 'scale(1.24' in tr or 'translate(50' in tr:
                children = list(elem)
                if not children:
                    continue
                child_tags = {strip_ns(c.tag) for c in children}
                elem.attrib['fill'] = '#A78BFA'
                if 'use' in child_tags:
                    c3 = elem
                elif 'g' in child_tags or 'path' in child_tags:
                    if op == '1' and c1 is None:
                        c1 = elem
                    elif op == '0' and c2 is None:
                        c2 = elem
                    
    if c1 is None or c2 is None:
        print("Error: Could not locate main containers!")
        return
        
    # Process Container #1 (Fade-In): Round-Robin interleaving across all child groups (60 fade-in layers)
    c1_child_paths = []
    for child in c1:
        if strip_ns(child.tag) == 'g':
            for gc in child:
                if strip_ns(gc.tag) == 'path':
                    c1_child_paths.append(gc)
                    break
                    
    num_c1_groups = len(c1_child_paths)
    print(f"Container #1 (Fade-In): Distributing {len(new_runs)} runs across {num_c1_groups} layers via Round-Robin...")
    c1_assigned = [[] for _ in range(num_c1_groups)]
    for idx, r in enumerate(new_runs):
        c1_assigned[idx % num_c1_groups].append(r)
        
    for i, p in enumerate(c1_child_paths):
        p.set("d", serialize_runs(c1_assigned[i]))
        
    # Process Container #2 (Logo Morph): Spatial Nearest-Neighbor Assignment for smooth logo transformation!
    c2_child_paths = []
    c2_centroids = []
    for child in c2:
        if strip_ns(child.tag) == 'g':
            for gc in child:
                if strip_ns(gc.tag) == 'path':
                    c2_child_paths.append(gc)
                    orig_runs = parse_runs_centroids(gc.attrib.get('d', ''))
                    if orig_runs:
                        cx = np.mean([x + l/2.0 for x, y, l in orig_runs])
                        cy = np.mean([y for x, y, l in orig_runs])
                    else:
                        cx, cy = 150, 150
                    c2_centroids.append((cx, cy))
                    break
                    
    num_c2_groups = len(c2_child_paths)
    print(f"Container #2 (Logo Morph): Matching {len(new_runs)} runs to {num_c2_groups} spatial transformation trajectories...")
    c2_assigned = [[] for _ in range(num_c2_groups)]
    
    c2_cents = np.array(c2_centroids)
    run_coords = np.array([(r[3], r[1]) for r in new_runs])
    
    dists = np.sum((run_coords[:, None, :] - c2_cents[None, :, :])**2, axis=-1)
    closest_groups = np.argmin(dists, axis=-1)
    
    for idx, grp_idx in enumerate(closest_groups):
        c2_assigned[grp_idx].append(new_runs[idx])
        
    for grp_idx in range(num_c2_groups):
        if len(c2_assigned[grp_idx]) == 0 and len(new_runs) > num_c2_groups:
            largest = np.argmax([len(lst) for lst in c2_assigned])
            if len(c2_assigned[largest]) > 1:
                c2_assigned[grp_idx].append(c2_assigned[largest].pop())
                
    for i, p in enumerate(c2_child_paths):
        p.set("d", serialize_runs(c2_assigned[i]))
        
    # Process Container #3 (Symbol Morph Sequence): Replace Triangle with Z and align start/end with new portrait!
    if c3 is not None:
        dots = list(c3)
        num_dots = len(dots)
        print(f"Container #3 (Symbol Morph): Updating {num_dots} dots -> changing Triangle icon into letter Z...")
        
        # Ensure we have exactly num_dots coordinates for Z
        curr_z_coords = z_coords if len(z_coords) == num_dots else generate_z_coordinates(num_dots)
        
        # Extract existing < / > coordinates (at index 4 of values) to match transitions cleanly
        slash_coords = []
        anim_transforms = []
        for dot in dots:
            anim_tr = None
            for sc in dot:
                if strip_ns(sc.tag) == 'animateTransform':
                    anim_tr = sc
                    break
            anim_transforms.append(anim_tr)
            if anim_tr is not None:
                vals = anim_tr.attrib.get('values', '').split(';')
                if len(vals) >= 5:
                    try:
                        sx, sy = map(float, vals[4].split())
                        slash_coords.append((sx, sy))
                    except:
                        slash_coords.append((150.0, 175.0))
                else:
                    slash_coords.append((150.0, 175.0))
            else:
                slash_coords.append((150.0, 175.0))
                
        # Perform smooth 1-to-1 spatial matching from < / > symbol to letter Z!
        matched_z_pts = match_points_1to1(slash_coords, curr_z_coords)
        
        # Perform matching from new portrait dot coordinates for clean start/end loop alignment
        portrait_pool = [(r[3], r[1]) for r in new_runs]
        if len(portrait_pool) < num_dots:
            portrait_pool.extend(portrait_pool[:num_dots - len(portrait_pool)])
        elif len(portrait_pool) > num_dots:
            idx_keep = np.linspace(0, len(portrait_pool)-1, num_dots).astype(int)
            portrait_pool = [portrait_pool[i] for i in idx_keep]
        matched_portrait_pts = match_points_1to1(slash_coords, portrait_pool)
        
        for i, anim_tr in enumerate(anim_transforms):
            if anim_tr is not None:
                vals = anim_tr.attrib.get('values', '').split(';')
                if len(vals) == 9:
                    zx, zy = int(round(matched_z_pts[i][0])), int(round(matched_z_pts[i][1]))
                    px, py = int(round(matched_portrait_pts[i][0])), int(round(matched_portrait_pts[i][1]))
                    z_str = f"{zx} {zy}"
                    p_str = f"{px} {py}"
                    
                    # Update Portrait start (0, 1), Triangle->Z slot (6, 7), and Portrait return (8)
                    vals[0] = p_str
                    vals[1] = p_str
                    vals[6] = z_str
                    vals[7] = z_str
                    vals[8] = p_str
                    anim_tr.attrib['values'] = ';'.join(vals)
    else:
        print("Warning: Container #3 (Morph sequence dots) not found!")
        
    # Register default namespace to prevent ns0: tags
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    tree.write(output_path, encoding='utf-8', xml_declaration=False)
    
    with open(output_path, 'r', encoding='utf-8') as f:
        svg_content = f.read()
    if 'ns0:' in svg_content or 'xmlns:ns0=' in svg_content:
        svg_content = svg_content.replace('ns0:', '').replace('xmlns:ns0=', 'xmlns=')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
            
    print(f"Successfully rebuilt {output_path} with Z morph target and uniform dark theme interior!")

def main():
    photo_path = "my image.png"
    if not os.path.exists(photo_path):
        print(f"Error: {photo_path} not found.")
        return
        
    print("Generating High-Definition Halftone Portraits for Dark and Light themes...")
    res_arr = load_and_crop_photo(photo_path, 300, 307)
    
    binary_dark = generate_halftone_dither(res_arr, is_dark_mode=True, target_density=0.182)
    runs_dark = binary_to_runs_with_coords(binary_dark)
    print(f"Dark Theme Portrait -> {binary_dark.sum()} active dots ({binary_dark.mean():.2%}), {len(runs_dark)} path runs.")
    
    # Use exact same high-fidelity facial dither as dark mode so face identity never flips or changes in light mode!
    binary_light = generate_halftone_dither(res_arr, is_dark_mode=True, target_density=0.182)
    runs_light = binary_to_runs_with_coords(binary_light)
    print(f"Light Theme Portrait (Identical Face) -> {binary_light.sum()} active dots ({binary_light.mean():.2%}), {len(runs_light)} path runs.")
    
    print("Generating geometry for letter 'Z' morph target...")
    z_coords = generate_z_coordinates(count=900)
    
    src_dark = "arifhaxn-main/dark_original_backup.svg" if os.path.exists("arifhaxn-main/dark_original_backup.svg") else "arifhaxn-main/dark.svg"
    src_light = "arifhaxn-main/light_original_backup.svg" if os.path.exists("arifhaxn-main/light_original_backup.svg") else "arifhaxn-main/light.svg"
    
    process_and_rebuild_svg(src_dark, "arifhaxn-main/dark.svg", runs_dark, True, z_coords)
    process_and_rebuild_svg(src_light, "arifhaxn-main/light.svg", runs_light, False, z_coords)
    
    # Save high-res previews
    h, w = binary_dark.shape
    dark_preview = np.full((h, w, 3), [10, 16, 31], dtype=np.uint8)
    dark_preview[binary_dark == 1] = [167, 139, 250]
    Image.fromarray(dark_preview).save("perfect_dark.png")
    
    light_preview = np.full((h, w, 3), [10, 16, 31], dtype=np.uint8)
    light_preview[binary_light == 1] = [167, 139, 250]
    Image.fromarray(light_preview).save("perfect_light.png")
    
    artifacts_dir = "C:/Users/M.Shahzaib/.gemini/antigravity/brain/d4cc5c92-ddf6-4f25-9f64-bcabfa9266d5"
    if os.path.exists(artifacts_dir):
        shutil.copyfile("perfect_dark.png", os.path.join(artifacts_dir, "perfect_dark.png"))
        shutil.copyfile("perfect_light.png", os.path.join(artifacts_dir, "perfect_light.png"))
    print("\nMaster build complete! Your face is identical in dark and light themes, interior frame stays dark, and animation transforms into letter Z instead of Triangle!")

if __name__ == "__main__":
    main()
