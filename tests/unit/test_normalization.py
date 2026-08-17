import pytest

from docintel.documents.normalization import (
    InvalidFilenameError,
    normalize_text,
    sanitize_filename,
)


def test_normalization_is_deterministic_and_minimal() -> None:
    source = "Cafe\u0301\r\nline  \r\x00\x01\n\n\n\nend\t "
    assert normalize_text(source) == "Café\nline\n\n\nend"
    assert normalize_text(source) == normalize_text(source)


def test_safe_filename_removes_paths_and_controls() -> None:
    assert sanitize_filename("../../folder\\report\x00.pdf") == "report.pdf"


def test_safe_filename_normalizes_unicode_and_length() -> None:
    assert sanitize_filename("re\u0301sume\u0301.pdf") == "résumé.pdf"
    long_filename = sanitize_filename(f"{'a' * 300}.pdf")
    assert long_filename.endswith(".pdf")
    assert len(long_filename) == 255


@pytest.mark.parametrize("filename", ["", "\x00", "..", "folder/"])
def test_safe_filename_rejects_empty_or_unsafe_names(filename: str) -> None:
    with pytest.raises(InvalidFilenameError):
        sanitize_filename(filename)
