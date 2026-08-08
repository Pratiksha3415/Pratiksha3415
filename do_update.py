"""
Backs up dark.svg/light.svg then swaps in dark_new.svg/light_new.svg.
See note in deploy_new_svgs.py — not needed for the current main pipeline.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

with open(ROOT / 'dark.svg', 'r', encoding='utf-8') as f:
    orig_dark = f.read()
with open(ROOT / 'dark_original_backup.svg', 'w', encoding='utf-8') as f:
    f.write(orig_dark)

with open(ROOT / 'dark_new.svg', 'r', encoding='utf-8') as f:
    new_dark = f.read()
with open(ROOT / 'dark.svg', 'w', encoding='utf-8') as f:
    f.write(new_dark)

with open(ROOT / 'light.svg', 'r', encoding='utf-8') as f:
    orig_light = f.read()
with open(ROOT / 'light_original_backup.svg', 'w', encoding='utf-8') as f:
    f.write(orig_light)

with open(ROOT / 'light_new.svg', 'r', encoding='utf-8') as f:
    new_light = f.read()
with open(ROOT / 'light.svg', 'w', encoding='utf-8') as f:
    f.write(new_light)

with open(ROOT / 'update_status.txt', 'w', encoding='utf-8') as f:
    f.write("UPDATE_COMPLETE_SUCCESSFULLY\n")

print("Update complete.")
