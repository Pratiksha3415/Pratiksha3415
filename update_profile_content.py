import os
import xml.etree.ElementTree as ET

def strip_ns(tag):
    return tag.split('}')[-1] if '}' in tag else tag

def update_svg_profile(file_path):
    print(f"--- Updating {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    tree = ET.parse(file_path)
    root = tree.getroot()

    # 1. Update Root aria-label
    if 'aria-label' in root.attrib:
        root.attrib['aria-label'] = "Muhammad Shahzaib Wajid — profile.sh --live"

    table_data = {
        '162': ('Subject', 'Muhammad Shahzaib Wajid', 14.0),
        '185': ('Role', 'Software Engineering Student | Flutter Developer | AI Automation Developer', 14.0),
        '208': ('Origin', 'Islamabad, Pakistan', 14.0),
        '231': ('Education', 'BS Software Engineering, Iqra University Islamabad (H-9 Campus)', 14.0),
        '254': ('Status', 'Building AI Products, Learning Full Stack Development, Open to Internship Opportunities', 11.5),
        '277': ('Company', 'Founder, Zaphtra Ltd.', 14.0),
        '308': ('ToolChain', 'Flutter, Dart, Java, JavaScript, HTML, CSS, Git, GitHub, VS Code, Android Studio, Figma, Firebase, SQLite, Groq API, REST API', 7.6),
        '331': ('Core.Lang', 'Dart, Java, JavaScript, C++', 14.0),
        '354': ('Core.Frontend', 'Flutter, HTML, CSS, JavaScript', 14.0),
        '377': ('Core.Backend', 'Firebase, REST API', 14.0),
        '400': ('Core.Database', 'SQLite, Firebase', 14.0),
        '454': ('Grid.Mail', 'm.shahzaibwajid0647@gmail.com', 14.0),
        '477': ('Grid.Portfolio', 'coming soon', 14.0),
        '500': ('Grid.LinkedIn', 'muhammad-shahzaib-wajid', 14.0),
        '523': ('Grid.GitHub', '@shahzaibwajid', 14.0),
        '546': ('Grid.Facebook', '@shahzaibwajid', 14.0),
    }

    for elem in root.iter():
        tag = strip_ns(elem.tag)
        if tag == 'text':
            y = elem.attrib.get('y')
            if y in ('29', '29.0'):
                elem.text = "m.shahzaibwajid0647@gmail.com - % ./profile.sh --live"
                print(f"Updated top title bar (y={y})")
            elif y == '136':
                elem.text = "m.shahzaibwajid0647@gmail.com"
                print(f"Updated header mail (y={y})")
            elif y in table_data:
                key_label, val_text, fsize = table_data[y]
                tspans = [child for child in elem if strip_ns(child.tag) == 'tspan']
                if len(tspans) >= 3:
                    key_str = f"{key_label} "
                    val_str = f" {val_text}"
                    
                    if fsize == 14.0:
                        needed_dots = 79 - len(key_str) - len(val_str)
                        if needed_dots < 1:
                            needed_dots = 1
                    else:
                        # For condensed lines, 2 dots provide clean alignment
                        needed_dots = 2
                        
                    dots_str = "." * needed_dots
                    
                    tspans[0].text = key_str
                    tspans[1].text = dots_str
                    tspans[2].text = val_str
                    
                    if fsize != 14.0:
                        elem.attrib['font-size'] = str(fsize)
                    elif 'font-size' in elem.attrib:
                        # Ensure default row font-size is restored if previously modified
                        elem.attrib['font-size'] = "14"
                        
                    print(f"Updated row y={y:<4} -> {key_str}{dots_str}{val_str} (font-size={fsize})")
                else:
                    print(f"Warning: y={y} has fewer than 3 tspans ({len(tspans)})")

    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    tree.write(file_path, encoding='utf-8', xml_declaration=False)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    if 'ns0:' in content or 'xmlns:ns0=' in content:
        content = content.replace('ns0:', '').replace('xmlns:ns0=', 'xmlns=')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    print(f"Successfully finalized {file_path}!\n")

if __name__ == '__main__':
    update_svg_profile('arifhaxn-main/dark.svg')
    update_svg_profile('arifhaxn-main/light.svg')
