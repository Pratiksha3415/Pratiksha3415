import shutil
import os

# Backup originals
shutil.copy2('arifhaxn-main/dark.svg', 'arifhaxn-main/dark_original_backup.svg')
shutil.copy2('arifhaxn-main/light.svg', 'arifhaxn-main/light_original_backup.svg')
print("Backed up original SVGs as dark_original_backup.svg and light_original_backup.svg.")

# Overwrite dark.svg and light.svg with new versions
shutil.move('arifhaxn-main/dark_new.svg', 'arifhaxn-main/dark.svg')
shutil.move('arifhaxn-main/light_new.svg', 'arifhaxn-main/light.svg')
print("Successfully updated dark.svg and light.svg with your new photo portrait!")
