from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET


DOCX = Path(r"D:\tianchi\新建 DOCX 文档.docx")
OUT = Path(r"D:\tianchi\rules_extracted.txt")

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def node_text(node):
    parts = []
    for child in node.iter():
        tag = child.tag
        if tag == f"{{{NS['w']}}}t" and child.text:
            parts.append(child.text)
        elif tag == f"{{{NS['w']}}}tab":
            parts.append("\t")
        elif tag == f"{{{NS['w']}}}br":
            parts.append("\n")
    return "".join(parts).strip()


def paragraph_text(paragraph):
    return node_text(paragraph)


def table_text(table):
    lines = []
    for row in table.findall(".//w:tr", NS):
        cells = []
        for cell in row.findall("./w:tc", NS):
            cell_parts = []
            for paragraph in cell.findall(".//w:p", NS):
                text = paragraph_text(paragraph)
                if text:
                    cell_parts.append(text)
            cells.append(" / ".join(cell_parts))
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines).strip()


def main():
    with ZipFile(DOCX) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    body = root.find("w:body", NS)

    chunks = []
    for element in list(body):
        if element.tag == f"{{{NS['w']}}}p":
            text = paragraph_text(element)
            if text:
                chunks.append(text)
        elif element.tag == f"{{{NS['w']}}}tbl":
            text = table_text(element)
            if text:
                chunks.append("[TABLE]\n" + text + "\n[/TABLE]")

    OUT.write_text("\n\n".join(chunks), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
