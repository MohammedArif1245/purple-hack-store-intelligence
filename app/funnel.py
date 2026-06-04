"""
Store Intelligence — Funnel Endpoint

GET /stores/{id}/funnel
Groups events by session. Counts sessions reaching each funnel stage:
ENTRY → any ZONE_ENTER → BILLING_QUEUE_JOIN → purchase (converted=true)
Computes drop-off % between stages.
Re-entries are collapsed into the original session — no double-counting.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, SessionRecord
from app.models import FunnelResponse, FunnelStage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["funnel"])


@router.get("/stores/{store_id}/funnel", response_model=FunnelResponse)
async def get_funnel(
    store_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Conversion funnel: ENTRY → ZONE_ENTER → BILLING → PURCHASE.
    
    Re-entries collapsed into original session — no double counting.
    Drop-off % between each stage.
    """
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Get all sessions for this store today
    query = select(SessionRecord).where(
        and_(
            SessionRecord.store_id == store_id,
            SessionRecord.entry_time >= today_start,
        )
    )
    result = await db.execute(query)
    sessions = result.scalars().all()

    # Count sessions at each funnel stage
    total_entries = len(sessions)
    zone_visitors = sum(
        1 for s in sessions
        if s.zones_visited and len(s.zones_visited) > 0
    )
    billing_visitors = sum(
        1 for s in sessions
        if s.zones_visited and "BILLING" in (s.zones_visited or [])
    )
    purchasers = sum(1 for s in sessions if s.converted)

    # Build funnel stages with drop-off percentages
    stages = []

    stages.append(FunnelStage(
        stage="Entry",
        count=total_entries,
        drop_off_pct=0.0,
    ))

    stages.append(FunnelStage(
        stage="Zone Visit",
        count=zone_visitors,
        drop_off_pct=_drop_off(total_entries, zone_visitors),
    ))

    stages.append(FunnelStage(
        stage="Billing Queue",
        count=billing_visitors,
        drop_off_pct=_drop_off(zone_visitors, billing_visitors),
    ))

    stages.append(FunnelStage(
        stage="Purchase",
        count=purchasers,
        drop_off_pct=_drop_off(billing_visitors, purchasers),
    ))

    return FunnelResponse(
        store_id=store_id,
        timestamp=now,
        stages=stages,
        total_sessions=total_entries,
    )


def _drop_off(prev: int, current: int) -> float:
    """Calculate drop-off percentage between funnel stages."""
    if prev == 0:
        return 0.0
    return round((1 - current / prev) * 100, 1)
