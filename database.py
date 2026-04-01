#!/usr/bin/env python3
"""
database.py - 資料庫模型與連線設定
"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import enum

DATABASE_URL = "postgresql://dentalvis:dentalvis2026@localhost:5432/dentalvis"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AnalysisType(str, enum.Enum):
    init   = "init"
    plaque = "plaque"

class AnalysisStatus(str, enum.Enum):
    queued  = "queued"
    running = "running"
    done    = "done"
    failed  = "failed"

class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String(255), unique=True, index=True, nullable=False)
    name          = Column(String(100), nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at    = Column(DateTime, default=lambda: __import__("datetime").datetime.now(__import__("datetime").timezone(__import__("datetime").timedelta(hours=8))).replace(tzinfo=None))
    analyses      = relationship("Analysis", back_populates="user", cascade="all, delete-orphan")

class Analysis(Base):
    __tablename__  = "analyses"
    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    type           = Column(Enum(AnalysisType), nullable=False)
    status         = Column(Enum(AnalysisStatus), default=AnalysisStatus.queued)
    created_at     = Column(DateTime, default=lambda: __import__("datetime").datetime.now(__import__("datetime").timezone(__import__("datetime").timedelta(hours=8))).replace(tzinfo=None))
    completed_at   = Column(DateTime, nullable=True)
    data_dir       = Column(String(512), nullable=True)   # 該次分析的資料夾
    result_json    = Column(Text, nullable=True)           # 結果 JSON 字串
    error_msg      = Column(Text, nullable=True)
    user           = relationship("User", back_populates="analyses")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")

if __name__ == "__main__":
    init_db()