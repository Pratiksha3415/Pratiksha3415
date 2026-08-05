# Read original files and backup cleanly via string read/write (avoiding Windows metadata ACL/timestamp copies)

with open('arifhaxn-main/dark.svg', 'r', encoding='utf-8') as f:
    orig_dark = f.read()
with open('arifhaxn-main/dark_original_backup.svg', 'w', encoding='utf-8') as f:
    f.write(orig_dark)

with open('arifhaxn-main/dark_new.svg', 'r', encoding='utf-8') as f:
    new_dark = f.read()
with open('arifhaxn-main/dark.svg', 'w', encoding='utf-8') as f:
    f.write(new_dark)

with open('arifhaxn-main/light.svg', 'r', encoding='utf-8') as f:
    orig_light = f.read()
with open('arifhaxn-main/light_original_backup.svg', 'w', encoding='utf-8') as f:
    f.write(orig_light)

with open('arifhaxn-main/light_new.svg', 'r', encoding='utf-8') as f:
    new_light = f.read()
with open('arifhaxn-main/light.svg', 'w', encoding='utf-8') as f:
    f.write(new_light)

with open('update_status.txt', 'w', encoding='utf-8') as f:
    f.write("UPDATE_COMPLETE_SUCCESSFULLY\n")
