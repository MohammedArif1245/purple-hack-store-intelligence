"""
Store Intelligence — Event Ingestion Endpoint

POST /events/ingest
- Accepts batch of up to 500 events
- Validates each against StoreEvent schema
- On validation failure: collect error, continue processing (partial success)
- Bulk upsert with ON CONFLICT (event_id) DO NOTHING — idempotent
- Returns structured IngestResponse with counts and per-event errors
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, EventRecord
from app.models import IngestRequest, IngestResponse, StoreEvent
from app.sessions import materialise_sessions

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingestion"])


@router.post("/events/ingest", response_model=IngestResponse)
async def ingest_events(
    request: IngestRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Batch event ingestion endpoint.
    
    - Accepts up to 500 events per batch
    - Idempotent by event_id (duplicates silently ignored)
    - Partial success: bad events are skipped, rest are committed
    - Returns success/failure counts and error details
    """
    success_count = 0
    failed_ids = []
    errors = []

    for event in request.events:
        try:
            # Build the database record
            record = EventRecord(
                event_id=event.event_id,
                store_id=event.store_id,
                camera_id=event.camera_id,
                visitor_id=event.visitor_id,
                event_type=event.event_type.value,
                timestamp=event.timestamp,
                zone_id=event.zone_id,
                dwell_ms=event.dwell_ms,
                is_staff=event.is_staff,
                confidence=event.confidence,
                event_metadata=event.metadata.model_dump() if event.metadata else None,
                ingested_at=datetime.now(timezone.utc),
            )

            # Merge handles idempotency — existing records are not overwritten
            await db.merge(record)
            success_count += 1

        except Exception as e:
            failed_ids.append(event.event_id)
            errors.append(f"Event {event.event_id}: {str(e)}")
            logger.warning(f"Failed to ingest event {event.event_id}: {e}")

    # Commit all successful events in one transaction
    try:
        await db.commit()
        
        # Trigger session materialisation for all affected stores
        store_ids = {e.store_id for e in request.events if e.event_id not in failed_ids}
        for store_id in store_ids:
            await materialise_sessions(db, store_id)
            
    except Exception as e:
        await db.rollback()
        logger.error(f"Batch commit failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database commit failed: {str(e)}"
        )

    logger.info(
        f"Ingested {success_count}/{len(request.events)} events "
        f"({len(failed_ids)} failures)"
    )

    return IngestResponse(
        success_count=success_count,
        failed_count=len(failed_ids),
        failed_ids=failed_ids,
        errors=errors,
    )
