import json
from pathlib import Path

FIXTURES_DIRECTORY = Path(__file__).parent / "fixtures"


def load_fixture_pages(filename: str) -> list[str]:
    fixture_path = FIXTURES_DIRECTORY / f"{Path(filename).stem}.json"
    pages = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(pages, list) or not all(isinstance(page, str) for page in pages):
        raise ValueError(f"Invalid PDF page fixture: {fixture_path.name}")
    return pages


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_text_pdf(page_texts: list[str], *, run_marker: str) -> bytes:
    """Build the controlled textual PDF corpus without a PDF-generation dependency."""
    page_count = len(page_texts)
    font_object_number = 3 + (2 * page_count)
    kids = " ".join(f"{3 + (2 * index)} 0 R" for index in range(page_count))
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode(),
        font_object_number: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    for index, text in enumerate(page_texts):
        page_object_number = 3 + (2 * index)
        content_object_number = page_object_number + 1
        objects[page_object_number] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_object_number} 0 R >> >> "
            f"/Contents {content_object_number} 0 R >>"
        ).encode()
        escaped = _escape_pdf_text(text)
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
        objects[content_object_number] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number in range(1, font_object_number + 1):
        offsets.append(len(output))
        output.extend(f"{object_number} 0 obj\n".encode())
        output.extend(objects[object_number])
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {font_object_number + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        (
            f"trailer\n<< /Size {font_object_number + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n% eval-run {run_marker}\n"
        ).encode()
    )
    return bytes(output)
