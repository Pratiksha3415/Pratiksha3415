"""
Swaps in freshly generated dark_new.svg / light_new.svg over the live
dark.svg / light.svg, keeping a backup of the previous versions.

NOTE: rebuild_perfect_svgs_with_morph.py (the main generator) now writes
directly to dark.svg/light.svg and doesn't produce *_new.svg intermediates,
so this script is only needed if you have a workflow that still produces
dark_new.svg/light_new.svg separately.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

dark_svg = ROOT / "dark.svg"
light_svg = ROOT / "light.svg"
dark_new = ROOT / "dark_new.svg"
light_new = ROOT / "light_new.svg"
dark_backup = ROOT / "dark_original_backup.svg"
light_backup = ROOT / "light_original_backup.svg"

if dark_new.exists():
    shutil.copy2(dark_svg, dark_backup)
    shutil.move(str(dark_new), str(dark_svg))
    print("Backed up dark.svg -> dark_original_backup.svg and applied dark_new.svg.")
else:
    print("No dark_new.svg found, skipping.")

if light_new.exists():
    shutil.copy2(light_svg, light_backup)
    shutil.move(str(light_new), str(light_svg))
    print("Backed up light.svg -> light_original_backup.svg and applied light_new.svg.")
else:
    print("No light_new.svg found, skipping.")

print("Done.")
