import uuid

import pytest

from docintel.providers.generation import GenerationOutput
from docintel.rag.sources import (
    CANONICAL_ABSTENTION_ANSWER,
    SourceValidationError,
    map_retrieval_sources,
    validate_grounded_output,
)
from docintel.retrieval.service import RetrievalResult


def result(rank: int) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        document_id=uuid.UUID(int=rank),
        filename=f"document-{rank}.pdf",
        page_number=rank,
        chunk_id=uuid.UUID(int=rank + 10),
        chunk_index=rank - 1,
        content=f"source {rank}",
        similarity=1.0 - (rank / 10),
    )


def test_one_source_maps_to_s1_with_real_metadata() -> None:
    retrieval = result(1)
    sources = map_retrieval_sources([retrieval])
    assert len(sources) == 1
    assert sources[0].source_id == "S1"
    assert sources[0].document_id == retrieval.document_id
    assert sources[0].filename == retrieval.filename
    assert sources[0].page_number == retrieval.page_number
    assert sources[0].chunk_id == retrieval.chunk_id
    assert sources[0].chunk_index == retrieval.chunk_index
    assert sources[0].text == retrieval.content
    assert sources[0].similarity == retrieval.similarity


def test_four_sources_map_in_retrieval_order() -> None:
    sources = map_retrieval_sources([result(rank) for rank in range(1, 5)])
    assert [source.source_id for source in sources] == ["S1", "S2", "S3", "S4"]
    assert [source.text for source in sources] == [
        "source 1",
        "source 2",
        "source 3",
        "source 4",
    ]


@pytest.mark.parametrize(
    "output",
    [
        GenerationOutput(answer="Supported [S1]", abstained=False, citation_ids=["S1"]),
        GenerationOutput(
            answer="First [S1], third [S3].",
            abstained=False,
            citation_ids=["S1", "S3"],
        ),
        GenerationOutput(
            answer="Repeated [S1] and [S1].",
            abstained=False,
            citation_ids=["S1"],
        ),
    ],
)
def test_valid_grounded_answers_are_accepted(output: GenerationOutput) -> None:
    sources = map_retrieval_sources([result(rank) for rank in range(1, 5)])
    validated = validate_grounded_output(output, sources)
    assert not validated.abstained
    assert validated.answer == output.answer
    assert validated.citation_ids == tuple(output.citation_ids)


@pytest.mark.parametrize(
    "output",
    [
        GenerationOutput(answer="Invented [S9]", abstained=False, citation_ids=["S9"]),
        GenerationOutput(answer="Known [S1]", abstained=False, citation_ids=["S9"]),
        GenerationOutput(
            answer="Duplicate [S1]",
            abstained=False,
            citation_ids=["S1", "S1"],
        ),
        GenerationOutput(answer="Mismatch [S2]", abstained=False, citation_ids=["S1"]),
        GenerationOutput(answer="No citation", abstained=False, citation_ids=["S1"]),
        GenerationOutput(answer="", abstained=False, citation_ids=[]),
        GenerationOutput(answer="", abstained=True, citation_ids=["S1"]),
        GenerationOutput(answer="Factual answer", abstained=True, citation_ids=[]),
    ],
)
def test_invalid_grounding_contract_is_rejected(output: GenerationOutput) -> None:
    sources = map_retrieval_sources([result(1), result(2)])
    with pytest.raises(SourceValidationError):
        validate_grounded_output(output, sources)


def test_valid_abstention_is_canonicalized() -> None:
    validated = validate_grounded_output(
        GenerationOutput(answer="  ", abstained=True, citation_ids=[]),
        map_retrieval_sources([result(1)]),
    )
    assert validated.abstained
    assert validated.answer == CANONICAL_ABSTENTION_ANSWER
    assert validated.citation_ids == ()
