import re

def verify(orig_path, new_path):
    with open(orig_path, 'r', encoding='utf-8') as f:
        orig_lines = f.read().split('\n')
    with open(new_path, 'r', encoding='utf-8') as f:
        new_lines = f.read().split('\n')
        
    print(f"\n=================== Verifying {orig_path} vs {new_path} ===================")
    if len(orig_lines) != len(new_lines):
        print(f"ERROR: Line count mismatch! Orig: {len(orig_lines)}, New: {len(new_lines)}")
        return
        
    modified_count = 0
    identical_count = 0
    structural_error_count = 0
    
    for idx, (old, new) in enumerate(zip(orig_lines, new_lines)):
        if old == new:
            identical_count += 1
        else:
            modified_count += 1
            # Check if difference is purely inside d="..."
            old_no_d = re.sub(r'd="[^"]*"', 'd="HIDDEN"', old)
            new_no_d = re.sub(r'd="[^"]*"', 'd="HIDDEN"', new)
            
            if old_no_d != new_no_d:
                print(f"STRUCTURAL MISMATCH on line {idx+1}:")
                print(f"  Old: {old[:100]}...")
                print(f"  New: {new[:100]}...")
                structural_error_count += 1
            else:
                # Let's ensure the line is strictly within the portrait region (lines 33 to 200)
                if idx < 30 or idx > 200:
                    print(f"WARNING: Modification outside expected portrait lines! Line {idx+1}")
                    
    print(f"Summary -> Total lines: {len(orig_lines)}, Identical lines: {identical_count}, Modified portrait path lines: {modified_count}")
    if structural_error_count == 0:
        print("SUCCESS: 100% structural integrity! All animations, attributes, colors, and layout are identically preserved.")
    else:
        print(f"FAILED: Found {structural_error_count} structural errors.")

verify('arifhaxn-main/dark.svg', 'arifhaxn-main/dark_new.svg')
verify('arifhaxn-main/light.svg', 'arifhaxn-main/light_new.svg')
