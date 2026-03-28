# database/db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'database', 'dr_records.db')

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at: {DB_PATH}")