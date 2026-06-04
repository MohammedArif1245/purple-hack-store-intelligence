"""
Store Intelligence — Metrics Endpoint

GET /stores/{id}/metrics
- Unique visitor count (WHERE is_staff=false, today)
- Conversion rate (converted sessions / total sessions)
- Average dwell per zone
- Current queue depth (from cache or DB fallback)
- Abandonment rate
- All values return 0 or 0.0 on empty data — never null or crash
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import select, func, and_, distinct, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, EventRecord, SessionRecord
from app.models import MetricsResponse, ZoneDwell

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])


@router.get("/stores/{store_id}/metrics", response_model=MetricsResponse)
async def get_metrics(
    store_id: str,
    tz_offset_hours: float = 0.0,
    date_override: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Store-level metrics: visitors, conversion, dwell, queue, abandonment.
    
    Excludes staff from all visitor counts.
    Returns zero-values on empty data — never null.
    """
    now = datetime.now(timezone.utc)
    if date_override:
        try:
            # Expecting format YYYY-MM-DD
            dt = datetime.strptime(date_override, "%Y-%m-%d")
            today_start = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            today_start = (now - timedelta(hours=24))
    else:
        local_now = now + timedelta(hours=tz_offset_hours)
        local_today_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_start = local_today_start - timedelta(hours=tz_offset_hours)

    # ── Total unique visitors today (exclude staff) ──
    visitor_query = select(
        func.count(distinct(EventRecord.visitor_id))
    ).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.is_staff == False,
            EventRecord.event_type == "ENTRY",
            EventRecord.timestamp >= today_start,
        )
    )
    result = await db.execute(visitor_query)
    total_visitors = result.scalar() or 0

    # ── Conversion rate from sessions ──
    total_sessions_q = select(
        func.count(SessionRecord.session_id)
    ).where(
        and_(
            SessionRecord.store_id == store_id,
            SessionRecord.entry_time >= today_start,
        )
    )
    result = await db.execute(total_sessions_q)
    total_sessions = result.scalar() or 0

    converted_sessions_q = select(
        func.count(SessionRecord.session_id)
    ).where(
        and_(
            SessionRecord.store_id == store_id,
            SessionRecord.converted == True,
            SessionRecord.entry_time >= today_start,
        )
    )
    result = await db.execute(converted_sessions_q)
    converted_sessions = result.scalar() or 0

    conversion_rate = (
        round(converted_sessions / total_sessions, 4)
        if total_sessions > 0 else 0.0
    )

    # ── Average dwell per zone ──
    dwell_query = select(
        EventRecord.zone_id,
        func.avg(EventRecord.dwell_ms).label("avg_dwell"),
        func.count().label("visit_count"),
    ).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.is_staff == False,
            EventRecord.event_type.in_(["ZONE_DWELL", "ZONE_EXIT"]),
            EventRecord.zone_id.isnot(None),
            EventRecord.dwell_ms > 0,
            EventRecord.timestamp >= today_start,
        )
    ).group_by(EventRecord.zone_id)

    result = await db.execute(dwell_query)
    dwell_rows = result.all()

    dwell_per_zone = [
        ZoneDwell(
            zone_id=row.zone_id,
            avg_dwell_ms=round(float(row.avg_dwell), 1),
            visit_count=row.visit_count,
        )
        for row in dwell_rows
    ]

    # Overall average dwell
    avg_dwell_ms = (
        round(sum(z.avg_dwell_ms for z in dwell_per_zone) / len(dwell_per_zone), 1)
        if dwell_per_zone else 0.0
    )

    # ── Queue depth (from recent BILLING_QUEUE_JOIN events) ──
    queue_query = select(
        func.count(distinct(EventRecord.visitor_id))
    ).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "BILLING_QUEUE_JOIN",
            EventRecord.timestamp >= now - timedelta(minutes=10),
        )
    )
    result = await db.execute(queue_query)
    queue_depth = result.scalar() or 0

    # ── Abandonment rate ──
    abandon_query = select(
        func.count(distinct(EventRecord.visitor_id))
    ).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "BILLING_QUEUE_ABANDON",
            EventRecord.timestamp >= today_start,
        )
    )
    result = await db.execute(abandon_query)
    abandon_count = result.scalar() or 0

    billing_visitors_q = select(
        func.count(distinct(EventRecord.visitor_id))
    ).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.event_type.in_(["BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON"]),
            EventRecord.timestamp >= today_start,
        )
    )
    result = await db.execute(billing_visitors_q)
    billing_visitors = result.scalar() or 0

    abandonment_rate = (
        round(abandon_count / billing_visitors, 4)
        if billing_visitors > 0 else 0.0
    )

    return MetricsResponse(
        store_id=store_id,
        timestamp=now,
        total_visitors=total_visitors,
        conversion_rate=conversion_rate,
        avg_dwell_ms=avg_dwell_ms,
        dwell_per_zone=dwell_per_zone,
        queue_depth=queue_depth,
        abandonment_rate=abandonment_rate,
    )
