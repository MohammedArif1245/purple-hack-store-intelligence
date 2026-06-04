"""
Store Intelligence — Anomaly Detection Endpoint

GET /stores/{id}/anomalies

Three detectors:
- BILLING_QUEUE_SPIKE: current queue depth > 1.5× 7-day average for this hour
- CONVERSION_DROP: today's conversion rate < 70% of 7-day rolling average
- DEAD_ZONE: any zone with zero ZONE_ENTER events in last 30 minutes

Each anomaly includes severity (INFO/WARN/CRITICAL) and suggested_action string.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import select, func, and_, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, EventRecord, SessionRecord
from app.models import (
    AnomalyResponse, AnomalyDetail, AnomalyType, Severity
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["anomalies"])


@router.get("/stores/{store_id}/anomalies", response_model=AnomalyResponse)
async def get_anomalies(
    store_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Detect operational anomalies for a store.
    
    Checks for queue spikes, conversion drops, and dead zones.
    Returns severity-tagged anomalies with actionable suggestions.
    """
    now = datetime.now(timezone.utc)
    anomalies = []

    # ── Detector 1: BILLING_QUEUE_SPIKE ──
    queue_anomaly = await _check_queue_spike(db, store_id, now)
    if queue_anomaly:
        anomalies.append(queue_anomaly)

    # ── Detector 2: CONVERSION_DROP ──
    conversion_anomaly = await _check_conversion_drop(db, store_id, now)
    if conversion_anomaly:
        anomalies.append(conversion_anomaly)

    # ── Detector 3: DEAD_ZONE ──
    dead_zones = await _check_dead_zones(db, store_id, now)
    anomalies.extend(dead_zones)

    return AnomalyResponse(
        store_id=store_id,
        timestamp=now,
        anomalies=anomalies,
        anomaly_count=len(anomalies),
    )


async def _check_queue_spike(
    db: AsyncSession, store_id: str, now: datetime
) -> AnomalyDetail | None:
    """
    BILLING_QUEUE_SPIKE: current queue depth > 1.5× 7-day average for this hour.
    """
    # Current queue depth (last 10 minutes)
    current_q = select(
        func.count(distinct(EventRecord.visitor_id))
    ).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "BILLING_QUEUE_JOIN",
            EventRecord.timestamp >= now - timedelta(minutes=10),
        )
    )
    result = await db.execute(current_q)
    current_depth = result.scalar() or 0

    # 7-day average for this hour
    seven_days_ago = now - timedelta(days=7)
    current_hour = now.hour

    avg_q = select(
        func.count(distinct(EventRecord.visitor_id))
    ).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "BILLING_QUEUE_JOIN",
            EventRecord.timestamp >= seven_days_ago,
            func.extract('hour', EventRecord.timestamp) == current_hour,
        )
    )
    result = await db.execute(avg_q)
    total_7day = result.scalar() or 0
    avg_depth = total_7day / 7 if total_7day > 0 else 0

    if avg_depth > 0 and current_depth > 1.5 * avg_depth:
        severity = Severity.CRITICAL if current_depth > 3 * avg_depth else Severity.WARN
        return AnomalyDetail(
            anomaly_type=AnomalyType.BILLING_QUEUE_SPIKE,
            severity=severity,
            description=(
                f"Queue depth ({current_depth}) is "
                f"{current_depth/avg_depth:.1f}× the 7-day average ({avg_depth:.1f})"
            ),
            suggested_action="Open additional checkout counters immediately",
            detected_at=now,
            details={
                "current_depth": current_depth,
                "avg_depth_7day": round(avg_depth, 1),
                "ratio": round(current_depth / avg_depth, 2),
            },
        )

    return None


async def _check_conversion_drop(
    db: AsyncSession, store_id: str, now: datetime
) -> AnomalyDetail | None:
    """
    CONVERSION_DROP: recent (2-hour) conversion rate < 70% of 7-day rolling average.
    """
    recent_start = now - timedelta(hours=2)

    # Recent conversion (last 2 hours)
    recent_total_q = select(
        func.count(SessionRecord.session_id)
    ).where(
        and_(
            SessionRecord.store_id == store_id,
            SessionRecord.entry_time >= recent_start,
        )
    )
    result = await db.execute(recent_total_q)
    total_recent = result.scalar() or 0

    recent_converted_q = select(
        func.count(SessionRecord.session_id)
    ).where(
        and_(
            SessionRecord.store_id == store_id,
            SessionRecord.entry_time >= recent_start,
            SessionRecord.converted == True,
        )
    )
    result = await db.execute(recent_converted_q)
    converted_recent = result.scalar() or 0

    recent_rate = converted_recent / total_recent if total_recent > 0 else 0

    # 7-day average conversion
    seven_days_ago = now - timedelta(days=7)
    avg_total_q = select(
        func.count(SessionRecord.session_id)
    ).where(
        and_(
            SessionRecord.store_id == store_id,
            SessionRecord.entry_time >= seven_days_ago,
            SessionRecord.entry_time < recent_start,
        )
    )
    result = await db.execute(avg_total_q)
    total_7day = result.scalar() or 0

    avg_converted_q = select(
        func.count(SessionRecord.session_id)
    ).where(
        and_(
            SessionRecord.store_id == store_id,
            SessionRecord.entry_time >= seven_days_ago,
            SessionRecord.entry_time < recent_start,
            SessionRecord.converted == True,
        )
    )
    result = await db.execute(avg_converted_q)
    converted_7day = result.scalar() or 0

    avg_rate = converted_7day / total_7day if total_7day > 0 else 0

    # Only flag if we have enough recent traffic to be statistically relevant
    if total_recent >= 5 and avg_rate > 0 and recent_rate < 0.7 * avg_rate:
        severity = Severity.CRITICAL if recent_rate < 0.5 * avg_rate else Severity.WARN
        return AnomalyDetail(
            anomaly_type=AnomalyType.CONVERSION_DROP,
            severity=severity,
            description=(
                f"Recent conversion rate ({recent_rate:.1%}) is below "
                f"70% of the 7-day average ({avg_rate:.1%})"
            ),
            suggested_action=(
                "Review store display, check for stockouts, "
                "and evaluate staff assistance coverage"
            ),
            detected_at=now,
            details={
                "recent_rate": round(recent_rate, 4),
                "avg_rate_7day": round(avg_rate, 4),
                "ratio": round(recent_rate / avg_rate, 2) if avg_rate > 0 else 0,
            },
        )

    return None


async def _check_dead_zones(
    db: AsyncSession, store_id: str, now: datetime
) -> list[AnomalyDetail]:
    """
    DEAD_ZONE: any zone with zero ZONE_ENTER events in last 30 minutes.
    """
    thirty_min_ago = now - timedelta(minutes=30)

    # Get all zones that had any activity today (last 24 hours)
    twenty_four_hours_ago = now - timedelta(hours=24)
    all_zones_q = select(
        distinct(EventRecord.zone_id)
    ).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.zone_id.isnot(None),
            EventRecord.event_type == "ZONE_ENTER",
            EventRecord.timestamp >= twenty_four_hours_ago,
        )
    )
    result = await db.execute(all_zones_q)
    all_zones = {row[0] for row in result.all()}

    # Get zones active in last 30 minutes
    active_zones_q = select(
        distinct(EventRecord.zone_id)
    ).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.zone_id.isnot(None),
            EventRecord.event_type == "ZONE_ENTER",
            EventRecord.timestamp >= thirty_min_ago,
        )
    )
    result = await db.execute(active_zones_q)
    active_zones = {row[0] for row in result.all()}

    dead_zones = all_zones - active_zones
    anomalies = []

    for zone_id in dead_zones:
        anomalies.append(AnomalyDetail(
            anomaly_type=AnomalyType.DEAD_ZONE,
            severity=Severity.INFO,
            description=f"Zone '{zone_id}' has had no visitors in the last 30 minutes",
            suggested_action=(
                f"Consider repositioning displays or signage "
                f"to drive traffic to {zone_id}"
            ),
            detected_at=now,
            details={"zone_id": zone_id, "inactive_minutes": 30},
        ))

    return anomalies
