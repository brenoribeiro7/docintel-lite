import os
from collections.abc import Iterator

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://docintel:docintel@localhost:5432/docintel",
)
os.environ.setdefault("APP_ENV", "test")

from docintel.db.models import Document  # noqa: E402
from docintel.db.session import SessionLocal  # noqa: E402


@pytest.fixture
def database_session() -> Iterator[Session]:
    with SessionLocal() as session:
        session.execute(delete(Document))
        session.commit()
        yield session
        session.rollback()
        session.execute(delete(Document))
        session.commit()
