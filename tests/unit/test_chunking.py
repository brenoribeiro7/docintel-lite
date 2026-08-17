from docintel.documents.extraction import ExtractedPage
from docintel.indexing.chunking import (
    CHUNK_ENCODING,
    CHUNK_OVERLAP_TOKENS,
    MAX_CHUNK_TOKENS,
    chunk_pages,
)
from tests.fixtures.tokens import ENCODING, text_with_token_count


def test_chunk_encoding_is_cl100k_base() -> None:
    assert CHUNK_ENCODING == "cl100k_base"
    assert ENCODING.name == CHUNK_ENCODING


def test_empty_page_produces_no_chunks() -> None:
    assert chunk_pages([ExtractedPage(page_number=1, content="")]) == []


def test_short_page_produces_one_chunk() -> None:
    chunks = chunk_pages([ExtractedPage(page_number=1, content="Short page")])
    assert [(chunk.page_number, chunk.chunk_index, chunk.content) for chunk in chunks] == [
        (1, 0, "Short page")
    ]


def test_exactly_600_tokens_produces_one_chunk() -> None:
    chunks = chunk_pages([ExtractedPage(1, text_with_token_count(600))])
    assert len(chunks) == 1
    assert chunks[0].token_count == MAX_CHUNK_TOKENS


def test_601_tokens_exposes_overlap_off_by_one() -> None:
    source_tokens = ENCODING.encode(text_with_token_count(601))
    chunks = chunk_pages([ExtractedPage(1, ENCODING.decode(source_tokens))])
    assert [chunk.token_count for chunk in chunks] == [600, 101]
    assert ENCODING.encode(chunks[1].content)[:100] == source_tokens[500:600]
    assert ENCODING.encode(chunks[1].content)[100:] == source_tokens[600:]


def test_large_paragraph_uses_token_windows() -> None:
    chunks = chunk_pages([ExtractedPage(1, text_with_token_count(1_600))])
    assert [chunk.token_count for chunk in chunks] == [600, 600, 600]


def test_multiple_paragraphs_are_grouped_while_they_fit() -> None:
    content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_pages([ExtractedPage(1, content)])
    assert len(chunks) == 1
    assert chunks[0].content == content


def test_paragraph_boundary_is_preferred_before_hard_limit() -> None:
    first = text_with_token_count(400).lstrip()
    second = text_with_token_count(400).lstrip()
    chunks = chunk_pages([ExtractedPage(1, f"{first}\n\n{second}")])
    assert len(chunks) == 2
    assert chunks[0].content.endswith("\n\n")
    assert chunks[0].token_count < MAX_CHUNK_TOKENS


def test_overlap_is_never_more_than_100_tokens() -> None:
    chunks = chunk_pages([ExtractedPage(1, text_with_token_count(2_100))])
    for previous, current in zip(chunks, chunks[1:], strict=False):
        previous_tokens = ENCODING.encode(previous.content)
        current_tokens = ENCODING.encode(current.content)
        assert current_tokens[:CHUNK_OVERLAP_TOKENS] == previous_tokens[-100:]


def test_chunks_never_cross_pages_and_indices_are_global() -> None:
    chunks = chunk_pages(
        [
            ExtractedPage(1, text_with_token_count(601)),
            ExtractedPage(2, text_with_token_count(601)),
        ]
    )
    assert [chunk.page_number for chunk in chunks] == [1, 1, 2, 2]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]
    page_two_tokens = ENCODING.encode(chunks[2].content)
    assert page_two_tokens == ENCODING.encode(text_with_token_count(601))[:600]


def test_no_chunk_exceeds_600_and_no_new_tokens_are_lost() -> None:
    original = ENCODING.encode(text_with_token_count(2_100))
    chunks = chunk_pages([ExtractedPage(1, ENCODING.decode(original))])
    reconstructed = ENCODING.encode(chunks[0].content)
    for chunk in chunks[1:]:
        reconstructed.extend(ENCODING.encode(chunk.content)[CHUNK_OVERLAP_TOKENS:])
    assert all(chunk.token_count <= MAX_CHUNK_TOKENS for chunk in chunks)
    assert reconstructed == original


def test_chunking_is_deterministic() -> None:
    pages = [ExtractedPage(1, text_with_token_count(1_101))]
    assert chunk_pages(pages) == chunk_pages(pages)


def test_unicode_boundaries_remain_valid() -> None:
    chunks = chunk_pages([ExtractedPage(1, "🙂 café 漢字 " * 1_000)])
    assert chunks
    assert all("�" not in chunk.content for chunk in chunks)
    assert all(chunk.content.encode("utf-8") for chunk in chunks)


def test_document_can_produce_exactly_250_chunks() -> None:
    chunks = chunk_pages([ExtractedPage(1, text_with_token_count(125_100))])
    assert len(chunks) == 250
    assert chunks[-1].chunk_index == 249
