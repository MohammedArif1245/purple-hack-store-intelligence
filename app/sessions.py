"""
Store Intelligence — Session Materialisation

Materialises raw events into visitor sessions. Groups by (store_id, visitor_id).
Session starts at ENTRY, ends at EXIT (or 2-hour timeout).
Re-entry: if REENTRY event found, extend existing session rather than new one.
Runs as background task triggered on each ingest batch.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, and_, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRecord, SessionRecord

logger = logging.getLogger(__name__)

SESSION_TIMEOUT_HOURS = 2


async def materialise_sessions(db: AsyncSession, store_id: str):
    """
    Build/update sessions from raw events for a given store.
    
    Groups events by visitor_id, creates sessions from ENTRY→EXIT pairs,
    and handles re-entry by extending existing sessions.
    """
    # Get all events for this store, ordered by time
    query = (
        select(EventRecord)
        .where(EventRecord.store_id == store_id)
        .order_by(EventRecord.timestamp.asc())
    )
    result = await db.execute(query)
    events = result.scalars().all()

    if not events:
        return

    # Group events by visitor_id
    visitor_events: dict[str, list[EventRecord]] = {}
    for event in events:
        if event.visitor_id not in visitor_events:
            visitor_events[event.visitor_id] = []
        visitor_events[event.visitor_id].append(event)

    sessions_created = 0
    sessions_updated = 0

    for visitor_id, v_events in visitor_events.items():
        # Skip staff
        if any(e.is_staff for e in v_events):
            continue

        current_session = None

        for event in v_events:
            if event.event_type == "ENTRY":
                # Check for existing open session (re-entry scenario)
                existing = await _find_open_session(db, store_id, visitor_id)

                if existing is None:
                    # Create new session
                    current_session = SessionRecord(
                        session_id=str(uuid.uuid4()),
                        store_id=store_id,
                        visitor_id=visitor_id,
                        entry_time=event.timestamp,
                        exit_time=None,
                        zones_visited=[],
                        converted=False,
                        is_reentry=False,
                    )
                    db.add(current_session)
                    sessions_created += 1
                else:
                    current_session = existing

            elif event.event_type == "REENTRY":
                # Extend existing session
                existing = await _find_open_session(db, store_id, visitor_id)
                if existing:
                    existing.is_reentry = True
                    current_session = existing
                    sessions_updated += 1
                else:
                    # Create new session marked as re-entry
                    current_session = SessionRecord(
                        session_id=str(uuid.uuid4()),
                        store_id=store_id,
                        visitor_id=visitor_id,
                        entry_time=event.timestamp,
                        exit_time=None,
                        zones_visited=[],
                        converted=False,
                        is_reentry=True,
                    )
                    db.add(current_session)
                    sessions_created += 1

            elif event.event_type == "EXIT":
                if current_session:
                    current_session.exit_time = event.timestamp

            elif event.event_type in ("ZONE_ENTER", "ZONE_DWELL"):
                if current_session and event.zone_id:
                    zones = current_session.zones_visited or []
                    if event.zone_id not in zones:
                        zones.append(event.zone_id)
                        current_session.zones_visited = zones

            elif event.event_type == "BILLING_QUEUE_JOIN":
                # Potential conversion — will be confirmed by POS correlation
                if current_session:
                    from app.database import POSTransaction
                    txn_query = select(POSTransaction).where(
                        and_(
                            POSTransaction.store_id == store_id,
                            POSTransaction.timestamp >= event.timestamp - timedelta(minutes=5),
                            POSTransaction.timestamp <= event.timestamp + timedelta(minutes=5)
                        )
                    ).limit(1)
                    res = await db.execute(txn_query)
                    txn = res.scalar_one_or_none()
                    if txn:
                        current_session.converted = True

    # Close timed-out sessions
    timeout_cutoff = datetime.now(timezone.utc) - timedelta(hours=SESSION_TIMEOUT_HOURS)
    timeout_query = (
        select(SessionRecord)
        .where(
            and_(
                SessionRecord.store_id == store_id,
                SessionRecord.exit_time.is_(None),
                SessionRecord.entry_time < timeout_cutoff,
            )
        )
    )
    result = await db.execute(timeout_query)
    timed_out = result.scalars().all()

    for session in timed_out:
        session.exit_time = session.entry_time + timedelta(hours=SESSION_TIMEOUT_HOURS)

    try:
        await db.commit()
        logger.info(
            f"Sessions materialised for {store_id}: "
            f"{sessions_created} created, {sessions_updated} updated, "
            f"{len(timed_out)} timed out"
        )
    except Exception as e:
        await db.rollback()
        logger.error(f"Session materialisation failed: {e}")


async def _find_open_session(
    db: AsyncSession, store_id: str, visitor_id: str
) -> SessionRecord | None:
    """Find an existing open session (no exit_time) for this visitor."""
    query = (
        select(SessionRecord)
        .where(
            and_(
                SessionRecord.store_id == store_id,
                SessionRecord.visitor_id == visitor_id,
                SessionRecord.exit_time.is_(None),
            )
        )
        .order_by(SessionRecord.entry_time.desc())
        .limit(1)
    )
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def mark_session_converted(
    db: AsyncSession, store_id: str, visitor_id: str
):
    """Mark a visitor's latest session as converted (purchase made)."""
    query = (
        select(SessionRecord)
        .where(
            and_(
                SessionRecord.store_id == store_id,
                SessionRecord.visitor_id == visitor_id,
            )
        )
        .order_by(SessionRecord.entry_time.desc())
        .limit(1)
    )
    result = await db.execute(query)
    session = result.scalar_one_or_none()

    if session:
        session.converted = True
        await db.commit()
        logger.info(f"Session {session.session_id} marked as converted")
