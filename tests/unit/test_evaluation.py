import uuid
from collections import Counter

from docintel.rag.service import RAGResult, RetrievalMetadata
from docintel.rag.sources import CANONICAL_ABSTENTION_ANSWER, GroundedSource
from evals.run import EvaluationCase, build_report, load_dataset


def source(document: str, page_number: int, position: int) -> GroundedSource:
    return GroundedSource(
        source_id=f"S{position}",
        document_id=uuid.UUID(int=position),
        filename=document,
        page_number=page_number,
        chunk_id=uuid.UUID(int=position + 10),
        chunk_index=page_number - 1,
        text="controlled evidence",
        similarity=1.0,
    )


def successful_result(case: EvaluationCase) -> RAGResult:
    if case.expected_abstention:
        return RAGResult(
            question=case.question,
            answer=CANONICAL_ABSTENTION_ANSWER,
            abstained=True,
            citation_ids=(),
            sources=(),
            retrieval=RetrievalMetadata(top_k=4, chunks_returned=0),
        )
    sources = tuple(
        source(evidence.document, evidence.page_number, position)
        for position, evidence in enumerate(case.expected_evidence, start=1)
    )
    answer = "; ".join(
        f"{fact} [S{min(position, len(sources))}]"
        for position, fact in enumerate(case.required_facts, start=1)
    )
    citation_ids = tuple(f"S{position}" for position in range(1, len(sources) + 1))
    return RAGResult(
        question=case.question,
        answer=answer,
        abstained=False,
        citation_ids=citation_ids,
        sources=sources,
        retrieval=RetrievalMetadata(top_k=4, chunks_returned=len(sources)),
    )


def successful_fixture() -> tuple[list[EvaluationCase], list[RAGResult]]:
    cases = load_dataset()
    return cases, [successful_result(case) for case in cases]


def test_dataset_has_exact_required_size_and_distribution() -> None:
    cases = load_dataset()
    assert len(cases) == 8
    assert Counter(case.type for case in cases) == {
        "answerable": 3,
        "multi_chunk": 1,
        "unanswerable": 2,
        "retrieval_trap": 1,
        "injection_like": 1,
    }


def test_successful_report_has_required_metrics() -> None:
    cases, results = successful_fixture()
    report = build_report(cases, results)
    assert report["passed"] is True
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    assert metrics == {
        "evidence_recall_at_4": 1.0,
        "evidence_hits": 7,
        "evidence_total": 7,
        "answerable_grounded_success": {"passed": 6, "total": 6},
        "unanswerable_abstention": {"passed": 2, "total": 2},
        "citation_validity": 1.0,
        "invalid_source_ids": 0,
        "retrieval_trap": "PASS",
        "injection_like": "PASS",
    }


def test_partial_multi_chunk_evidence_fails_recall() -> None:
    cases, results = successful_fixture()
    case_index = next(index for index, case in enumerate(cases) if case.id == "M1")
    original = results[case_index]
    results[case_index] = RAGResult(
        question=original.question,
        answer=original.answer,
        abstained=original.abstained,
        citation_ids=("S1",),
        sources=original.sources[:1],
        retrieval=RetrievalMetadata(top_k=4, chunks_returned=1),
    )
    report = build_report(cases, results)
    assert report["passed"] is False
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["evidence_recall_at_4"] < 1.0


def test_wrong_unanswerable_result_fails() -> None:
    cases, results = successful_fixture()
    case_index = next(index for index, case in enumerate(cases) if case.id == "U1")
    results[case_index] = RAGResult(
        question=cases[case_index].question,
        answer="A CEO [S1]",
        abstained=False,
        citation_ids=("S1",),
        sources=(source("operations_guide.pdf", 1, 1),),
        retrieval=RetrievalMetadata(top_k=4, chunks_returned=1),
    )
    assert build_report(cases, results)["passed"] is False


def test_invalid_citation_fails() -> None:
    cases, results = successful_fixture()
    results[0] = RAGResult(
        question=results[0].question,
        answer="8443 [S9]",
        abstained=False,
        citation_ids=("S9",),
        sources=results[0].sources,
        retrieval=results[0].retrieval,
    )
    report = build_report(cases, results)
    assert report["passed"] is False
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["invalid_source_ids"] == 1


def test_forbidden_fact_fails() -> None:
    cases, results = successful_fixture()
    trap_index = next(index for index, case in enumerate(cases) if case.id == "T1")
    results[trap_index] = RAGResult(
        question=results[trap_index].question,
        answer="60 seconds [S1]",
        abstained=False,
        citation_ids=("S1",),
        sources=results[trap_index].sources,
        retrieval=results[trap_index].retrieval,
    )
    report = build_report(cases, results)
    assert report["passed"] is False
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["retrieval_trap"] == "FAIL"


def test_missing_required_fact_fails_without_a_forbidden_fact() -> None:
    cases, results = successful_fixture()
    results[0] = RAGResult(
        question=results[0].question,
        answer="A supported but incomplete answer [S1]",
        abstained=False,
        citation_ids=("S1",),
        sources=results[0].sources,
        retrieval=results[0].retrieval,
    )
    assert build_report(cases, results)["passed"] is False


def test_compromised_injection_case_fails() -> None:
    cases, results = successful_fixture()
    injection_index = next(index for index, case in enumerate(cases) if case.id == "I1")
    results[injection_index] = RAGResult(
        question=results[injection_index].question,
        answer="COMPROMISED [S1]",
        abstained=False,
        citation_ids=("S1",),
        sources=results[injection_index].sources,
        retrieval=results[injection_index].retrieval,
    )
    report = build_report(cases, results)
    assert report["passed"] is False
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["injection_like"] == "FAIL"
