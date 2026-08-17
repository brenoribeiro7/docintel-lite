import bisect
import codecs
import re
from dataclasses import dataclass

import tiktoken

from docintel.documents.extraction import ExtractedPage

CHUNK_ENCODING = "cl100k_base"
MAX_CHUNK_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 100
MAX_CHUNKS_PER_DOCUMENT = 250


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    chunk_index: int
    page_number: int
    content: str
    token_count: int


@dataclass(frozen=True, slots=True)
class _TokenBoundaries:
    safe_positions: list[int]
    safe_character_offsets: list[int]


def _token_boundaries(encoding: tiktoken.Encoding, tokens: list[int]) -> _TokenBoundaries:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    safe_positions = [0]
    safe_character_offsets = [0]
    character_count = 0

    for position, token in enumerate(tokens, start=1):
        character_count += len(decoder.decode(encoding.decode_single_token_bytes(token)))
        if not decoder.getstate()[0]:
            safe_positions.append(position)
            safe_character_offsets.append(character_count)
    decoder.decode(b"", final=True)
    return _TokenBoundaries(safe_positions, safe_character_offsets)


def _paragraph_token_boundaries(content: str, boundaries: _TokenBoundaries) -> set[int]:
    positions: set[int] = set()
    for separator in re.finditer(r"\n{2,}", content):
        index = bisect.bisect_left(boundaries.safe_character_offsets, separator.end())
        if index < len(boundaries.safe_positions):
            positions.add(boundaries.safe_positions[index])
    return positions


def _previous_safe_position(safe_positions: list[int], target: int, minimum: int) -> int:
    index = bisect.bisect_right(safe_positions, target) - 1
    while index >= 0 and safe_positions[index] <= minimum:
        index -= 1
    if index < 0:
        raise RuntimeError("No safe token boundary can advance the chunk window.")
    return safe_positions[index]


def _next_window_start(safe_positions: list[int], *, chunk_start: int, chunk_end: int) -> int:
    minimum_start = max(chunk_start + 1, chunk_end - CHUNK_OVERLAP_TOKENS)
    index = bisect.bisect_left(safe_positions, minimum_start)
    if index >= len(safe_positions) or safe_positions[index] >= chunk_end:
        raise RuntimeError("No safe token boundary can preserve bounded overlap.")
    return safe_positions[index]


def chunk_pages(pages: list[ExtractedPage]) -> list[ChunkDraft]:
    encoding = tiktoken.get_encoding(CHUNK_ENCODING)
    chunks: list[ChunkDraft] = []

    for page in pages:
        if not page.content.strip():
            continue
        tokens = encoding.encode(page.content)
        boundaries = _token_boundaries(encoding, tokens)
        paragraph_boundaries = _paragraph_token_boundaries(page.content, boundaries)
        start = 0

        while start < len(tokens):
            hard_end = min(start + MAX_CHUNK_TOKENS, len(tokens))
            if hard_end == len(tokens):
                end = len(tokens)
            else:
                maximum_safe_end = _previous_safe_position(
                    boundaries.safe_positions,
                    hard_end,
                    start,
                )
                paragraph_candidates = [
                    position
                    for position in paragraph_boundaries
                    if start + CHUNK_OVERLAP_TOKENS < position <= maximum_safe_end
                ]
                end = max(paragraph_candidates, default=maximum_safe_end)

            content = encoding.decode(tokens[start:end], errors="strict")
            actual_token_count = len(encoding.encode(content))
            while actual_token_count > MAX_CHUNK_TOKENS:
                end = _previous_safe_position(boundaries.safe_positions, end - 1, start)
                content = encoding.decode(tokens[start:end], errors="strict")
                actual_token_count = len(encoding.encode(content))

            if content.strip():
                chunks.append(
                    ChunkDraft(
                        chunk_index=len(chunks),
                        page_number=page.page_number,
                        content=content,
                        token_count=actual_token_count,
                    )
                )
            if end == len(tokens):
                break
            start = _next_window_start(
                boundaries.safe_positions,
                chunk_start=start,
                chunk_end=end,
            )

    return chunks
