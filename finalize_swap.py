"""
Alternate swap-in helper for dark_new.svg/light_new.svg -> dark.svg/light.svg.
See note in deploy_new_svgs.py — the main pipeline (rebuild_perfect_svgs_with_morph.py)
no longer needs this step.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

print("Starting SVG portrait swap...")

dark_new = ROOT / "dark_new.svg"
dark_svg = ROOT / "dark.svg"
if dark_new.exists() and dark_svg.exists():
    shutil.copyfile(dark_svg, ROOT / "dark_original_backup.svg")
    shutil.copyfile(dark_new, dark_svg)
    print("Successfully backed up original dark.svg and replaced portrait!")

light_new = ROOT / "light_new.svg"
light_svg = ROOT / "light.svg"
if light_new.exists() and light_svg.exists():
    shutil.copyfile(light_svg, ROOT / "light_original_backup.svg")
    shutil.copyfile(light_new, light_svg)
    print("Successfully backed up original light.svg and replaced portrait!")

print("All tasks finished successfully!")
