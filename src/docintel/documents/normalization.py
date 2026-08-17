import unicodedata


class InvalidFilenameError(ValueError):
    """Raised when an upload filename cannot be represented safely."""


def normalize_text(text: str) -> str:
    """Apply the deliberately small, deterministic M1 normalization policy."""
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    filtered = "".join(
        character
        for character in normalized
        if character in {"\n", "\t"} or not unicodedata.category(character).startswith("C")
    )

    result: list[str] = []
    consecutive_empty_lines = 0
    for line in filtered.split("\n"):
        trimmed = line.rstrip()
        if trimmed:
            consecutive_empty_lines = 0
            result.append(trimmed)
        elif consecutive_empty_lines < 2:
            consecutive_empty_lines += 1
            result.append("")
    return "\n".join(result)


def sanitize_filename(untrusted_filename: str) -> str:
    """Return a display-only filename, never a filesystem path."""
    normalized = unicodedata.normalize("NFC", untrusted_filename).replace("\\", "/")
    basename = normalized.rsplit("/", maxsplit=1)[-1]
    cleaned = "".join(
        character for character in basename if not unicodedata.category(character).startswith("C")
    ).strip()

    if not cleaned or cleaned in {".", ".."}:
        raise InvalidFilenameError("The upload filename is empty or unsafe.")

    if len(cleaned) > 255:
        suffix = ".pdf" if cleaned.casefold().endswith(".pdf") else ""
        cleaned = f"{cleaned[: 255 - len(suffix)]}{suffix}"
    return cleaned
