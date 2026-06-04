"""
Store Intelligence — Database Layer

SQLAlchemy 2.0 async engine with dual-mode support:
- SQLite + aiosqlite for local development (zero setup)
- PostgreSQL + asyncpg for production/Docker deployment

Auto-detects mode from DATABASE_URL environment variable.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text,
    JSON, Index, event, text
)
from sqlalchemy.ext.asyncio import (
    create_async_engine, async_sessionmaker, AsyncSession
)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)


# ─── Database URL Resolution ────────────────────────────────────────────────

def get_database_url() -> str:
    """
    Resolve database URL from environment.
    Falls back to SQLite for local development.
    """
    url = os.environ.get("DATABASE_URL", "")
    
    if url:
        # Convert postgres:// to postgresql+asyncpg://
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url
    
    # Default: SQLite for local development
    return "sqlite+aiosqlite:///./store_intelligence.db"


# ─── SQLAlchemy Base ─────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─── Table Definitions ───────────────────────────────────────────────────────

class EventRecord(Base):
    """Raw events from detection pipeline."""
    __tablename__ = "events"

    event_id = Column(String(50), primary_key=True)
    store_id = Column(String(50), nullable=False, index=True)
    camera_id = Column(String(50))
    visitor_id = Column(String(50), nullable=False, index=True)
    event_type = Column(String(30), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    zone_id = Column(String(50))
    dwell_ms = Column(Integer, default=0)
    is_staff = Column(Boolean, default=False)
    confidence = Column(Float, nullable=False)
    event_metadata = Column("metadata", JSON)
    ingested_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("idx_events_store_ts", "store_id", "timestamp"),
        Index("idx_events_visitor", "visitor_id"),
        Index("idx_events_type", "event_type"),
    )


class SessionRecord(Base):
    """Materialised visitor sessions."""
    __tablename__ = "sessions"

    session_id = Column(String(50), primary_key=True)
    store_id = Column(String(50), nullable=False, index=True)
    visitor_id = Column(String(50), nullable=False, index=True)
    entry_time = Column(DateTime(timezone=True))
    exit_time = Column(DateTime(timezone=True))
    zones_visited = Column(JSON, default=list)  # Store as JSON array
    converted = Column(Boolean, default=False)
    is_reentry = Column(Boolean, default=False)


class POSTransaction(Base):
    """POS transactions loaded from CSV."""
    __tablename__ = "pos_transactions"

    transaction_id = Column(String(50), primary_key=True)
    store_id = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    basket_value_inr = Column(Float, default=0.0)


# ─── Engine & Session Factory ────────────────────────────────────────────────

_engine = None
_session_factory = None


def get_engine():
    """Get or create the async engine (singleton)."""
    global _engine
    if _engine is None:
        url = get_database_url()
        is_sqlite = "sqlite" in url

        connect_args = {}
        if is_sqlite:
            connect_args["check_same_thread"] = False

        _engine = create_async_engine(
            url,
            echo=False,
            connect_args=connect_args,
            pool_pre_ping=True if not is_sqlite else False,
        )
        logger.info(f"Database engine created: {'SQLite' if is_sqlite else 'PostgreSQL'}")

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
    return _session_factory


async def get_db() -> AsyncSession:
    """FastAPI dependency: yields an async database session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_database():
    """Create all tables. Called on application startup."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created successfully")


async def close_database():
    """Dispose of the engine. Called on application shutdown."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
    logger.info("Database connection closed")


async def load_pos_transactions():
    """Load POS transactions from CSV into the database on startup."""
    import csv
    import os
    from sqlalchemy import select
    
    csv_path = "pos_transactions.csv"
    if not os.path.exists(csv_path):
        logger.warning(f"POS CSV not found: {csv_path}")
        return
        
    factory = get_session_factory()
    
    async with factory() as session:
        # Check if already loaded to avoid duplicates
        result = await session.execute(select(POSTransaction).limit(1))
        if result.scalar_one_or_none() is not None:
            logger.info("POS transactions already loaded.")
            return

        records = []
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(POSTransaction(
                        transaction_id=row.get("transaction_id", ""),
                        store_id=row.get("store_id", ""),
                        timestamp=datetime.fromisoformat(row.get("timestamp", "").replace("Z", "+00:00")),
                        basket_value_inr=float(row.get("basket_value_inr", 0))
                    ))
            
            if records:
                session.add_all(records)
                await session.commit()
                logger.info(f"Loaded {len(records)} POS transactions into database.")
        except Exception as e:
            logger.error(f"Failed to load POS transactions: {e}")
            await session.rollback()
