"""
Store Intelligence — Heatmap Endpoint

GET /stores/{id}/heatmap
Zone frequency, avg dwell, normalised score 0-100, data confidence.
Flags LOW confidence if < 20 sessions in the window.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, EventRecord
from app.models import HeatmapResponse, HeatmapZone

logger = logging.getLogger(__name__)

router = APIRouter(tags=["heatmap"])


@router.get("/stores/{store_id}/heatmap", response_model=HeatmapResponse)
async def get_heatmap(
    store_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Zone-level heatmap with visit frequency, dwell, and normalised scores.
    
    Each zone gets a score 0-100 relative to the busiest zone.
    Data confidence flagged LOW if < 20 sessions in the zone.
    """
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Get zone visit frequencies and dwell times
    query = select(
        EventRecord.zone_id,
        func.count().label("visit_freq"),
        func.avg(EventRecord.dwell_ms).label("avg_dwell_ms"),
    ).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.is_staff == False,
            EventRecord.zone_id.isnot(None),
            EventRecord.event_type.in_(["ZONE_ENTER", "ZONE_DWELL"]),
            EventRecord.timestamp >= today_start,
        )
    ).group_by(EventRecord.zone_id)

    result = await db.execute(query)
    rows = result.all()

    if not rows:
        return HeatmapResponse(
            store_id=store_id,
            timestamp=now,
            zones=[],
        )

    # Find max frequency for normalisation
    max_freq = max(row.visit_freq for row in rows)

    zones = []
    for row in rows:
        # Normalise score 0-100 relative to busiest zone
        score = round((row.visit_freq / max_freq) * 100, 1) if max_freq > 0 else 0.0
        
        # Flag low confidence if < 20 sessions
        data_confidence = "HIGH" if row.visit_freq >= 20 else "LOW"

        zones.append(HeatmapZone(
            zone_id=row.zone_id,
            visit_freq=row.visit_freq,
            avg_dwell_ms=round(float(row.avg_dwell_ms or 0), 1),
            score=score,
            data_confidence=data_confidence,
        ))

    # Sort by score descending
    zones.sort(key=lambda z: z.score, reverse=True)

    return HeatmapResponse(
        store_id=store_id,
        timestamp=now,
        zones=zones,
    )
