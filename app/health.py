"""
Store Intelligence — Health Check Endpoint

GET /health
Service status, last event per store, stale feed detection.
STALE_FEED if > 10 min lag per store.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import select, func, distinct, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, EventRecord
from app.models import HealthResponse, StoreHealth, ServiceStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Track when the app started
_start_time = time.time()

STALE_THRESHOLD_MINUTES = 10


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db: AsyncSession = Depends(get_db),
):
    """
    System health check.
    
    Verifies database connectivity, checks feed staleness per store,
    and reports overall system status.
    """
    now = datetime.now(timezone.utc)
    uptime = time.time() - _start_time

    # Check database connectivity
    db_status = "connected"
    try:
        await db.execute(select(func.count()).select_from(EventRecord))
    except Exception as e:
        db_status = f"error: {str(e)}"

    # Redis status (honest status)
    redis_status = {"status": "not_configured", "note": "Redis not yet integrated"}

    # Get per-store health
    stores_query = select(
        EventRecord.store_id,
        func.max(EventRecord.timestamp).label("last_event_at"),
        func.count().label("event_count"),
    ).group_by(EventRecord.store_id)

    result = await db.execute(stores_query)
    store_rows = result.all()

    stores = []
    stale_feeds = []

    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Get today's events for all stores in one query to avoid N+1
    today_counts_q = select(
        EventRecord.store_id,
        func.count().label("count")
    ).where(
        EventRecord.timestamp >= today_start
    ).group_by(EventRecord.store_id)
    today_counts_result = await db.execute(today_counts_q)
    today_counts = {row.store_id: row.count for row in today_counts_result.all()}

    for row in store_rows:
        last_event = row.last_event_at
        is_stale = False

        if last_event:
            # Make timezone-aware if needed
            if last_event.tzinfo is None:
                last_event = last_event.replace(tzinfo=timezone.utc)
            time_since = (now - last_event).total_seconds() / 60
            is_stale = time_since > STALE_THRESHOLD_MINUTES

        event_count_today = today_counts.get(row.store_id, 0)

        store_health = StoreHealth(
            store_id=row.store_id,
            last_event_at=last_event,
            is_stale=is_stale,
            event_count_today=event_count_today,
        )
        stores.append(store_health)

        if is_stale:
            stale_feeds.append(row.store_id)

    # Determine overall status
    if db_status != "connected":
        status = ServiceStatus.UNHEALTHY
    elif stale_feeds:
        status = ServiceStatus.DEGRADED
    else:
        status = ServiceStatus.HEALTHY

    return HealthResponse(
        status=status,
        database=db_status,
        redis=redis_status,
        uptime_seconds=round(uptime, 2),
        stores=stores,
        stale_feeds=stale_feeds,
    )
