import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from docintel.providers.generation import GenerationOutput
from docintel.retrieval.service import RetrievalResult

CANONICAL_ABSTENTION_ANSWER = (
    "Insufficient evidence in the selected documents to answer the question."
)
_INLINE_SOURCE_PATTERN = re.compile(r"\[(S[^\[\]\s]*)\]")


class SourceValidationError(Exception):
    """The structured generation output violates the grounding contract."""


@dataclass(frozen=True, slots=True)
class GroundedSource:
    source_id: str
    document_id: uuid.UUID
    filename: str
    page_number: int
    chunk_id: uuid.UUID
    chunk_index: int
    text: str
    similarity: float


@dataclass(frozen=True, slots=True)
class ValidatedGroundedAnswer:
    answer: str
    abstained: bool
    citation_ids: tuple[str, ...]


def map_retrieval_sources(results: Sequence[RetrievalResult]) -> list[GroundedSource]:
    return [
        GroundedSource(
            source_id=f"S{position}",
            document_id=result.document_id,
            filename=result.filename,
            page_number=result.page_number,
            chunk_id=result.chunk_id,
            chunk_index=result.chunk_index,
            text=result.content,
            similarity=result.similarity,
        )
        for position, result in enumerate(results, start=1)
    ]


def extract_inline_source_ids(answer: str) -> list[str]:
    return _INLINE_SOURCE_PATTERN.findall(answer)


def validate_grounded_output(
    output: GenerationOutput,
    sources: Sequence[GroundedSource],
) -> ValidatedGroundedAnswer:
    if output.abstained:
        if output.answer.strip() or output.citation_ids:
            raise SourceValidationError("An abstention cannot contain an answer or citations.")
        return ValidatedGroundedAnswer(
            answer=CANONICAL_ABSTENTION_ANSWER,
            abstained=True,
            citation_ids=(),
        )

    answer = output.answer.strip()
    if not answer or not output.citation_ids:
        raise SourceValidationError("A grounded answer requires text and citations.")
    if len(set(output.citation_ids)) != len(output.citation_ids):
        raise SourceValidationError("Citation IDs cannot contain duplicates.")

    valid_source_ids = {source.source_id for source in sources}
    declared_ids = set(output.citation_ids)
    if not declared_ids <= valid_source_ids:
        raise SourceValidationError("A citation ID was not supplied by retrieval.")

    inline_ids = extract_inline_source_ids(answer)
    if not inline_ids or not set(inline_ids) <= valid_source_ids:
        raise SourceValidationError("The answer contains an invalid inline citation.")
    if set(inline_ids) != declared_ids:
        raise SourceValidationError("Inline citations and citation_ids do not match.")

    return ValidatedGroundedAnswer(
        answer=answer,
        abstained=False,
        citation_ids=tuple(output.citation_ids),
    )
