import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

def extract_docx_text(docx_path: Path) -> str:
    if not docx_path.exists():
        raise FileNotFoundError(f"File not found: {docx_path}")

    with zipfile.ZipFile(docx_path, 'r') as zf:
        with zf.open('word/document.xml') as f:
            xml_bytes = f.read()

    # Parse XML
    ns = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    }
    root = ET.fromstring(xml_bytes)

    paragraphs = []
    for p in root.findall('.//w:p', ns):
        texts = []
        # Handle line breaks within a run
        for elem in p.iter():
            tag = elem.tag
            if isinstance(tag, str) and tag.endswith('}t'):
                if elem.text:
                    texts.append(elem.text)
            elif isinstance(tag, str) and tag.endswith('}br'):
                texts.append('\n')
        para = ''.join(texts).strip()
        paragraphs.append(para)

    # Remove empty consecutive lines while preserving section spacing
    cleaned = []
    prev_empty = False
    for para in paragraphs:
        if para:
            cleaned.append(para)
            prev_empty = False
        else:
            if not prev_empty:
                cleaned.append('')
            prev_empty = True

    return '\n\n'.join(cleaned).strip()


def main():
    if len(sys.argv) < 2:
        print('Usage: python extract_docx_text.py <path-to-docx>')
        sys.exit(1)
    path = Path(sys.argv[1])
    try:
        text = extract_docx_text(path)
    except KeyError as e:
        # Likely missing expected XML entries
        print(f'Error reading DOCX structure: {e}', file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(3)
    print(text)


if __name__ == '__main__':
    main()