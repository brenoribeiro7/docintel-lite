import argparse
import json
import sys
import uuid
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter
from sqlalchemy.orm import Session

from docintel.config import get_settings
from docintel.db.models import Document
from docintel.db.session import SessionLocal
from docintel.documents.service import ingest_document
from docintel.providers.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EmbeddingProviderFailure,
    OpenAIEmbeddingProvider,
)
from docintel.providers.generation import (
    GENERATION_MODEL,
    GenerationOutput,
    OpenAIGenerationProvider,
)
from docintel.rag.service import RAGResult, run_rag_query
from docintel.rag.sources import CANONICAL_ABSTENTION_ANSWER, extract_inline_source_ids
from evals.pdf_fixtures import load_fixture_pages, make_text_pdf

DATASET_PATH = Path(__file__).with_name("dataset.json")
CORPUS_FILENAMES = ("operations_guide.pdf", "security_guide.pdf")


class EvidenceLocation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document: str
    page_number: int


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    type: Literal[
        "answerable",
        "multi_chunk",
        "unanswerable",
        "retrieval_trap",
        "injection_like",
    ]
    question: str
    selected_documents: list[str]
    required_facts: list[str]
    expected_evidence: list[EvidenceLocation]
    forbidden_facts: list[str]
    expected_abstention: bool


def load_dataset(path: Path = DATASET_PATH) -> list[EvaluationCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TypeAdapter(list[EvaluationCase]).validate_python(payload)


def _unit_vector(index: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[index] = 1.0
    return vector


class DeterministicEvaluationEmbeddingProvider:
    model = EMBEDDING_MODEL
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, cases: Sequence[EvaluationCase]) -> None:
        page_texts = [
            page for filename in CORPUS_FILENAMES for page in load_fixture_pages(filename)
        ]
        self._page_vectors = {
            content: _unit_vector(index) for index, content in enumerate(page_texts)
        }
        self._question_vectors: dict[str, list[float]] = {}
        for case in cases:
            if case.id == "M1":
                vector = [0.0] * EMBEDDING_DIMENSIONS
                vector[2] = 1.0
                vector[3] = 1.0
            elif case.expected_evidence:
                evidence = case.expected_evidence[0]
                page_offset = 0 if evidence.document == "operations_guide.pdf" else 7
                vector = _unit_vector(page_offset + evidence.page_number - 1)
            else:
                vector = _unit_vector(20)
            self._question_vectors[case.question] = vector
        self.calls: list[list[str]] = []

    @property
    def is_configured(self) -> bool:
        return True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors: list[list[float]] = []
        for text in texts:
            vector = self._page_vectors.get(text) or self._question_vectors.get(text)
            if vector is None:
                raise EmbeddingProviderFailure("Unknown controlled evaluation text.")
            vectors.append(list(vector))
        return vectors


class DeterministicEvaluationGenerationProvider:
    model = GENERATION_MODEL

    def __init__(self, cases: Sequence[EvaluationCase]) -> None:
        self._cases = {case.question: case for case in cases}
        self.calls: list[dict[str, str]] = []

    @property
    def is_configured(self) -> bool:
        return True

    def generate(self, *, instructions: str, input_text: str) -> GenerationOutput:
        self.calls.append({"instructions": instructions, "input_text": input_text})
        payload = json.loads(input_text)
        question = payload["question"]
        case = self._cases[question]
        if case.expected_abstention:
            return GenerationOutput(answer="", abstained=True, citation_ids=[])

        answer_parts: list[str] = []
        citation_ids: list[str] = []
        for fact in case.required_facts:
            matching_source = next(
                (
                    source
                    for source in payload["sources"]
                    if fact.casefold() in source["content"].casefold()
                ),
                None,
            )
            if matching_source is None:
                return GenerationOutput(answer="", abstained=True, citation_ids=[])
            source_id = matching_source["source_id"]
            answer_parts.append(f"{fact} [{source_id}]")
            if source_id not in citation_ids:
                citation_ids.append(source_id)
        return GenerationOutput(
            answer="; ".join(answer_parts),
            abstained=False,
            citation_ids=citation_ids,
        )


def _case_evaluation(case: EvaluationCase, result: RAGResult) -> dict[str, object]:
    gold = {(item.document, item.page_number) for item in case.expected_evidence}
    retrieved = {(source.filename, source.page_number) for source in result.sources}
    evidence_hits = len(gold & retrieved)
    normalized_answer = result.answer.casefold()
    required_facts_present = all(
        fact.casefold() in normalized_answer for fact in case.required_facts
    )
    forbidden_facts_absent = all(
        fact.casefold() not in normalized_answer for fact in case.forbidden_facts
    )
    valid_source_ids = {source.source_id for source in result.sources}
    inline_ids = extract_inline_source_ids(result.answer)
    citation_valid = (
        len(set(result.citation_ids)) == len(result.citation_ids)
        and set(result.citation_ids) <= valid_source_ids
        and (
            (result.abstained and not result.citation_ids)
            or (
                not result.abstained
                and bool(result.citation_ids)
                and set(inline_ids) == set(result.citation_ids)
            )
        )
    )
    if case.expected_abstention:
        passed = (
            result.abstained
            and not result.citation_ids
            and result.answer == CANONICAL_ABSTENTION_ANSWER
            and citation_valid
        )
    else:
        passed = (
            not result.abstained
            and required_facts_present
            and forbidden_facts_absent
            and evidence_hits == len(gold)
            and citation_valid
        )
    return {
        "id": case.id,
        "type": case.type,
        "passed": passed,
        "abstained": result.abstained,
        "evidence_hits": evidence_hits,
        "evidence_total": len(gold),
        "citation_valid": citation_valid,
        "invalid_source_ids": len(set(result.citation_ids) - valid_source_ids),
        "required_facts_present": required_facts_present,
        "forbidden_facts_absent": forbidden_facts_absent,
    }


def _report_int(report: dict[str, object], key: str) -> int:
    value = report[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Evaluation report field {key} must be an integer.")
    return value


def build_report(
    cases: Sequence[EvaluationCase],
    results: Sequence[RAGResult],
) -> dict[str, object]:
    case_reports = [
        _case_evaluation(case, result) for case, result in zip(cases, results, strict=True)
    ]
    answerable = [
        report
        for case, report in zip(cases, case_reports, strict=True)
        if not case.expected_abstention
    ]
    unanswerable = [
        report for case, report in zip(cases, case_reports, strict=True) if case.expected_abstention
    ]
    evidence_hits = sum(_report_int(report, "evidence_hits") for report in answerable)
    evidence_total = sum(_report_int(report, "evidence_total") for report in answerable)
    citation_passes = sum(bool(report["citation_valid"]) for report in case_reports)
    invalid_source_ids = sum(_report_int(report, "invalid_source_ids") for report in case_reports)
    trap_pass = next(bool(report["passed"]) for report in case_reports if report["id"] == "T1")
    injection_pass = next(bool(report["passed"]) for report in case_reports if report["id"] == "I1")
    distribution = dict(sorted(Counter(case.type for case in cases).items()))
    metrics = {
        "evidence_recall_at_4": evidence_hits / evidence_total if evidence_total else 0.0,
        "evidence_hits": evidence_hits,
        "evidence_total": evidence_total,
        "answerable_grounded_success": {
            "passed": sum(bool(report["passed"]) for report in answerable),
            "total": len(answerable),
        },
        "unanswerable_abstention": {
            "passed": sum(bool(report["passed"]) for report in unanswerable),
            "total": len(unanswerable),
        },
        "citation_validity": citation_passes / len(case_reports),
        "invalid_source_ids": invalid_source_ids,
        "retrieval_trap": "PASS" if trap_pass else "FAIL",
        "injection_like": "PASS" if injection_pass else "FAIL",
    }
    passed = (
        len(cases) == 8
        and metrics["evidence_recall_at_4"] == 1.0
        and metrics["answerable_grounded_success"] == {"passed": 6, "total": 6}
        and metrics["unanswerable_abstention"] == {"passed": 2, "total": 2}
        and metrics["citation_validity"] == 1.0
        and invalid_source_ids == 0
        and trap_pass
        and injection_pass
    )
    return {
        "mode": "deterministic_or_live_contract",
        "dataset_cases": len(cases),
        "distribution": distribution,
        "metrics": metrics,
        "cases": case_reports,
        "passed": passed,
    }


def run_evaluation(
    session: Session,
    *,
    cases: Sequence[EvaluationCase],
    embedding_provider: DeterministicEvaluationEmbeddingProvider | OpenAIEmbeddingProvider,
    generation_provider: (DeterministicEvaluationGenerationProvider | OpenAIGenerationProvider),
) -> dict[str, object]:
    documents: dict[str, Document] = {}
    created_ids: list[uuid.UUID] = []
    run_marker = uuid.uuid4().hex
    try:
        for filename in CORPUS_FILENAMES:
            document = ingest_document(
                session,
                filename=filename,
                media_type="application/pdf",
                file_bytes=make_text_pdf(
                    load_fixture_pages(filename),
                    run_marker=f"{run_marker}-{filename}",
                ),
                embedding_provider=embedding_provider,
            )
            documents[filename] = document
            created_ids.append(document.id)

        results = [
            run_rag_query(
                session,
                question=case.question,
                document_ids=[documents[alias].id for alias in case.selected_documents],
                embedding_provider=embedding_provider,
                generation_provider=generation_provider,
            )
            for case in cases
        ]
        return build_report(cases, results)
    finally:
        session.rollback()
        for document_id in created_ids:
            stored_document = session.get(Document, document_id)
            if stored_document is not None:
                session.delete(stored_document)
        session.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the controlled DocIntel Lite evaluation.")
    parser.add_argument(
        "--mode",
        choices=("deterministic", "live"),
        default="deterministic",
        help="live performs paid OpenAI calls and is never intended for CI",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_dataset()
    embedding_provider: DeterministicEvaluationEmbeddingProvider | OpenAIEmbeddingProvider
    generation_provider: DeterministicEvaluationGenerationProvider | OpenAIGenerationProvider
    if args.mode == "live":
        settings = get_settings()
        if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
            print(
                json.dumps(
                    {"mode": "live", "passed": False, "error": "OPENAI_API_KEY is required."}
                )
            )
            return 2
        embedding_provider = OpenAIEmbeddingProvider(api_key=settings.openai_api_key)
        generation_provider = OpenAIGenerationProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_generation_model,
        )
    else:
        embedding_provider = DeterministicEvaluationEmbeddingProvider(cases)
        generation_provider = DeterministicEvaluationGenerationProvider(cases)

    with SessionLocal() as session:
        report = run_evaluation(
            session,
            cases=cases,
            embedding_provider=embedding_provider,
            generation_provider=generation_provider,
        )
    report["mode"] = args.mode
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
