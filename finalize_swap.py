import os
import shutil

print("Starting SVG portrait swap...")
if os.path.exists("arifhaxn-main/dark_new.svg") and os.path.exists("arifhaxn-main/dark.svg"):
    shutil.copyfile("arifhaxn-main/dark.svg", "arifhaxn-main/dark_original_backup.svg")
    shutil.copyfile("arifhaxn-main/dark_new.svg", "arifhaxn-main/dark.svg")
    print("Successfully backed up original dark.svg and replaced portrait!")

if os.path.exists("arifhaxn-main/light_new.svg") and os.path.exists("arifhaxn-main/light.svg"):
    shutil.copyfile("arifhaxn-main/light.svg", "arifhaxn-main/light_original_backup.svg")
    shutil.copyfile("arifhaxn-main/light_new.svg", "arifhaxn-main/light.svg")
    print("Successfully backed up original light.svg and replaced portrait!")

print("All tasks finished successfully!")
