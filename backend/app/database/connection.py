from pathlib import Path
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.core.config import settings
from backend.app.database.models import Base

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_database_url() -> str:
    db_url = os.getenv("DATABASE_URL", getattr(settings, "DATABASE_URL", "sqlite:///data/railsetu.db"))
    if db_url.startswith("sqlite:///"):
        rel_path = db_url.replace("sqlite:///", "")
        if not os.path.isabs(rel_path):
            full_path = REPO_ROOT / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{full_path}"
    return db_url


db_url = get_database_url()
engine = create_engine(
    db_url,
    connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(target_engine=None):
    use_engine = target_engine or engine
    Base.metadata.create_all(bind=use_engine)
