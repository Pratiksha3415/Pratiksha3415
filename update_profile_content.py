"""
Updates the identity/bio/links text rows baked into dark.svg and light.svg.

This does NOT touch the halftone portrait artwork or the SVG's animations/
structure — it only rewrites the <text>/<tspan> content and social <a href>
targets for the info-table rows (Subject, Role, Origin, Education, Status,
Company, ToolChain, Core.*, Grid.*).

Run from the repo root:
    python update_profile_content.py
"""

import os
import re
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
DARK_SVG = ROOT / "dark.svg"
LIGHT_SVG = ROOT / "light.svg"


def strip_ns(tag):
    return tag.split('}')[-1] if '}' in tag else tag


# Row data keyed by the *actual* y-coordinate of each <text> element in the
# SVG (verified against the current dark.svg/light.svg layout).
# Each value is (label, display_text, font_size).
TABLE_DATA = {
    '158': ('Subject', 'Pratiksha Aghav', 14.0),
    '178': ('Role', 'Computer Engineering Student | Full-Stack Developer | Backend Engineer', 14.0),
    '198': ('Origin', 'Pune, Maharashtra, India', 14.0),
    '218': ('Education', 'B.Tech Computer Engineering, AISSMS Institute of Information Technology, Pune', 11.5),
    '238': ('Status', 'Backend Engineer Intern at Flyrank AI, Full-Stack Developer', 14.0),
    '258': ('', 'Open to Internship & Full-Time Opportunities', 14.0),
    '278': ('Company', 'Backend Engineer Intern, Flyrank AI', 14.0),
    '300': ('ToolChain', 'Git, GitHub, VS Code, Postman, Docker, AWS, Socket.IO,', 7.6),
    '320': ('', 'Cloudinary, Selenium, BeautifulSoup, NumPy, Pandas, Matplotlib', 14.0),
    '340': ('Core.Lang', 'C, C++, Python, Java, JavaScript, TypeScript, SQL', 11.5),
    '360': ('Core.Frontend', 'HTML5, CSS3, Bootstrap, React.js, Tailwind CSS', 14.0),
    '380': ('Core.Backend', 'Node.js, Express.js, REST APIs', 14.0),
    '400': ('Core.Database', 'MongoDB, MySQL, PostgreSQL, Firebase', 14.0),
    '420': ('Core.Infra', 'Git, GitHub, Docker, AWS', 14.0),
    '466': ('Grid.Mail', 'pratiksha.aghav2027@gmail.com', 14.0),
    '487': ('Grid.Portfolio', 'Not available', 14.0),
    '508': ('Grid.LinkedIn', 'https://linkedin.com/in/pratiksha-aghav-a38bab28a/', 14.0),
    '529': ('Grid.GitHub', 'https://github.com/Pratiksha3415', 14.0),
    '550': ('Grid.Instagram', 'Not available', 14.0),
}

# href targets for the social/contact rows (row label -> new href).
# Rows with no real link on the resume point at "#" rather than a fake URL.
HREF_BY_LABEL = {
    'Grid.Mail': 'mailto:pratiksha.aghav2027@gmail.com',
    'Grid.Portfolio': '#',
    'Grid.LinkedIn': 'https://linkedin.com/in/pratiksha-aghav-a38bab28a/',
    'Grid.GitHub': 'https://github.com/Pratiksha3415',
    'Grid.Instagram': '#',
}

TOP_TITLE_BAR_TEXT = "pratiksha.aghav2027@gmail.com - % ./profile.sh --live"
HEADER_MAIL_TEXT = "pratiksha.aghav2027@gmail.com"
ARIA_LABEL = "Pratiksha Aghav — profile.sh --live"


def fix_hrefs(content: str) -> str:
    """Rewrite each social/contact row's <a href="..."> based on its Grid.* label."""
    for label, new_href in HREF_BY_LABEL.items():
        pattern = re.compile(
            r'(<a href=")[^"]*("[^>]*>(?:(?!</a>).)*?<tspan[^>]*>' + re.escape(label) + r' </tspan>)',
            re.DOTALL,
        )
        content = pattern.sub(lambda m: m.group(1) + new_href + m.group(2), content)
    return content


def update_svg_profile(file_path: Path):
    print(f"--- Updating {file_path.name} ---")
    if not file_path.exists():
        print(f"Error: {file_path} not found.")
        return

    tree = ET.parse(file_path)
    root = tree.getroot()

    if 'aria-label' in root.attrib:
        root.attrib['aria-label'] = ARIA_LABEL

    for elem in root.iter():
        tag = strip_ns(elem.tag)
        if tag != 'text':
            continue
        y = elem.attrib.get('y')
        if y in ('29', '29.0'):
            elem.text = TOP_TITLE_BAR_TEXT
            print(f"Updated top title bar (y={y})")
        elif y == '136':
            elem.text = HEADER_MAIL_TEXT
            print(f"Updated header mail (y={y})")
        elif y in TABLE_DATA:
            key_label, val_text, fsize = TABLE_DATA[y]
            tspans = [child for child in elem if strip_ns(child.tag) == 'tspan']
            if len(tspans) >= 3:
                key_str = f"{key_label} " if key_label else ""
                val_str = f" {val_text}"

                if fsize == 14.0:
                    needed_dots = 79 - len(key_str) - len(val_str)
                    if needed_dots < 1:
                        needed_dots = 1
                else:
                    needed_dots = 2

                dots_str = "." * needed_dots

                tspans[0].text = key_str
                tspans[1].text = dots_str
                tspans[2].text = val_str

                if fsize != 14.0:
                    elem.attrib['font-size'] = str(fsize)
                elif 'font-size' in elem.attrib:
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

    content = fix_hrefs(content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Successfully finalized {file_path.name}!\n")


if __name__ == '__main__':
    update_svg_profile(DARK_SVG)
    update_svg_profile(LIGHT_SVG)
