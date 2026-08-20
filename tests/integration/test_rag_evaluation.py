import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from docintel.db.models import Document
from evals.run import (
    DeterministicEvaluationEmbeddingProvider,
    DeterministicEvaluationGenerationProvider,
    load_dataset,
    run_evaluation,
)

pytestmark = pytest.mark.database


def test_full_deterministic_evaluation_uses_real_postgres_and_cleans_up(
    database_session: Session,
) -> None:
    cases = load_dataset()
    embedding_provider = DeterministicEvaluationEmbeddingProvider(cases)
    generation_provider = DeterministicEvaluationGenerationProvider(cases)
    report = run_evaluation(
        database_session,
        cases=cases,
        embedding_provider=embedding_provider,
        generation_provider=generation_provider,
    )
    assert report["passed"] is True
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["evidence_recall_at_4"] == 1.0
    assert metrics["answerable_grounded_success"] == {"passed": 6, "total": 6}
    assert metrics["unanswerable_abstention"] == {"passed": 2, "total": 2}
    assert metrics["retrieval_trap"] == "PASS"
    assert metrics["injection_like"] == "PASS"
    assert len(embedding_provider.calls) == 10
    assert len(generation_provider.calls) == 8
    database_session.expire_all()
    assert database_session.scalar(select(func.count(Document.id))) == 0
