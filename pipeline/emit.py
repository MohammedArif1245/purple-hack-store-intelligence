"""
Store Intelligence — Event Emitter

Validates detection pipeline outputs against the StoreEvent schema,
assigns UUIDv4 event_ids, derives ISO-8601 timestamps, and writes
validated events to JSONL files.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import StoreEvent, EventType, EventMetadata

logger = logging.getLogger(__name__)


class EventEmitter:
    """
    Builds, validates, and writes store events to JSONL files.
    
    Acts as the translator between AI model outputs and the backend system.
    Ensures all events conform to the StoreEvent schema before writing.
    """

    def __init__(self, output_dir: str = ".", store_id: str = "STORE_BLR_002"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.store_id = store_id
        self._event_count = 0
        self._error_count = 0
        self._output_file = self.output_dir / f"events_{store_id}.jsonl"

    def build_event(
        self,
        visitor_id: str,
        event_type: EventType,
        timestamp: datetime,
        camera_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        dwell_ms: int = 0,
        is_staff: bool = False,
        confidence: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> Optional[StoreEvent]:
        """
        Build and validate a single StoreEvent.
        
        Returns None on validation failure (logged, not raised).
        """
        try:
            event_data = {
                "event_id": str(uuid.uuid4()),
                "store_id": self.store_id,
                "camera_id": camera_id,
                "visitor_id": visitor_id,
                "event_type": event_type,
                "timestamp": timestamp,
                "zone_id": zone_id,
                "dwell_ms": dwell_ms,
                "is_staff": is_staff,
                "confidence": confidence,
            }

            if metadata:
                event_data["metadata"] = EventMetadata(**metadata)

            event = StoreEvent(**event_data)
            return event

        except ValidationError as e:
            self._error_count += 1
            logger.error(
                f"Event validation failed for visitor {visitor_id}: {e}"
            )
            return None
        except Exception as e:
            self._error_count += 1
            logger.error(
                f"Unexpected error building event for visitor {visitor_id}: {e}"
            )
            return None

    def emit(self, event: StoreEvent) -> bool:
        """
        Write a validated event to the JSONL output file.
        
        Returns True on success, False on failure.
        """
        try:
            event_json = event.model_dump(mode="json")
            
            with open(self._output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_json, default=str) + "\n")

            self._event_count += 1
            logger.debug(
                f"Emitted event {event.event_id} "
                f"({event.event_type.value}) for {event.visitor_id}"
            )
            return True

        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to write event {event.event_id}: {e}")
            return False

    def build_and_emit(self, **kwargs) -> bool:
        """Build, validate, and emit in one step."""
        event = self.build_event(**kwargs)
        if event is None:
            return False
        return self.emit(event)

    def emit_entry(
        self, visitor_id: str, timestamp: datetime,
        camera_id: str, confidence: float, is_staff: bool = False
    ) -> bool:
        """Convenience: emit an ENTRY event."""
        return self.build_and_emit(
            visitor_id=visitor_id,
            event_type=EventType.ENTRY,
            timestamp=timestamp,
            camera_id=camera_id,
            confidence=confidence,
            is_staff=is_staff,
            metadata={"session_seq": 1},
        )

    def emit_exit(
        self, visitor_id: str, timestamp: datetime,
        camera_id: str, confidence: float, session_seq: int = 1
    ) -> bool:
        """Convenience: emit an EXIT event."""
        return self.build_and_emit(
            visitor_id=visitor_id,
            event_type=EventType.EXIT,
            timestamp=timestamp,
            camera_id=camera_id,
            confidence=confidence,
            metadata={"session_seq": session_seq},
        )

    def emit_zone_enter(
        self, visitor_id: str, timestamp: datetime,
        camera_id: str, zone_id: str, confidence: float,
        session_seq: int = 1
    ) -> bool:
        """Convenience: emit a ZONE_ENTER event."""
        return self.build_and_emit(
            visitor_id=visitor_id,
            event_type=EventType.ZONE_ENTER,
            timestamp=timestamp,
            camera_id=camera_id,
            zone_id=zone_id,
            confidence=confidence,
            metadata={"session_seq": session_seq},
        )

    def emit_zone_dwell(
        self, visitor_id: str, timestamp: datetime,
        camera_id: str, zone_id: str, dwell_ms: int,
        confidence: float, session_seq: int = 1,
        sku_zone: Optional[str] = None
    ) -> bool:
        """Convenience: emit a ZONE_DWELL event (emitted every 30s of continuous containment)."""
        meta = {"session_seq": session_seq}
        if sku_zone:
            meta["sku_zone"] = sku_zone
        return self.build_and_emit(
            visitor_id=visitor_id,
            event_type=EventType.ZONE_DWELL,
            timestamp=timestamp,
            camera_id=camera_id,
            zone_id=zone_id,
            dwell_ms=dwell_ms,
            confidence=confidence,
            metadata=meta,
        )

    def emit_zone_exit(
        self, visitor_id: str, timestamp: datetime,
        camera_id: str, zone_id: str, dwell_ms: int,
        confidence: float, session_seq: int = 1
    ) -> bool:
        """Convenience: emit a ZONE_EXIT event."""
        return self.build_and_emit(
            visitor_id=visitor_id,
            event_type=EventType.ZONE_EXIT,
            timestamp=timestamp,
            camera_id=camera_id,
            zone_id=zone_id,
            dwell_ms=dwell_ms,
            confidence=confidence,
            metadata={"session_seq": session_seq},
        )

    def emit_reentry(
        self, visitor_id: str, timestamp: datetime,
        camera_id: str, confidence: float,
        reid_confidence: float, session_seq: int = 1
    ) -> bool:
        """Convenience: emit a REENTRY event (re-identified after prior exit)."""
        return self.build_and_emit(
            visitor_id=visitor_id,
            event_type=EventType.REENTRY,
            timestamp=timestamp,
            camera_id=camera_id,
            confidence=confidence,
            metadata={
                "reid_confidence": reid_confidence,
                "session_seq": session_seq,
            },
        )

    def emit_billing_queue_join(
        self, visitor_id: str, timestamp: datetime,
        camera_id: str, confidence: float,
        queue_depth: int, session_seq: int = 1
    ) -> bool:
        """Convenience: emit BILLING_QUEUE_JOIN."""
        return self.build_and_emit(
            visitor_id=visitor_id,
            event_type=EventType.BILLING_QUEUE_JOIN,
            timestamp=timestamp,
            camera_id=camera_id,
            zone_id="BILLING",
            confidence=confidence,
            metadata={
                "queue_depth": queue_depth,
                "session_seq": session_seq,
            },
        )

    def emit_billing_queue_abandon(
        self, visitor_id: str, timestamp: datetime,
        camera_id: str, confidence: float,
        dwell_ms: int, queue_depth: int, session_seq: int = 1
    ) -> bool:
        """Convenience: emit BILLING_QUEUE_ABANDON."""
        return self.build_and_emit(
            visitor_id=visitor_id,
            event_type=EventType.BILLING_QUEUE_ABANDON,
            timestamp=timestamp,
            camera_id=camera_id,
            zone_id="BILLING",
            dwell_ms=dwell_ms,
            confidence=confidence,
            metadata={
                "queue_depth": queue_depth,
                "session_seq": session_seq,
            },
        )

    def derive_timestamp(
        self,
        clip_start: datetime,
        frame_number: int,
        fps: float = 30.0
    ) -> datetime:
        """
        Derive event timestamp from clip start time + frame offset.
        
        Args:
            clip_start: When the video clip recording began (UTC)
            frame_number: Current frame index in the clip
            fps: Frames per second of the video
            
        Returns:
            UTC datetime for this specific frame
        """
        offset_ms = int((frame_number / fps) * 1000)
        return clip_start + timedelta(milliseconds=offset_ms)

    @property
    def stats(self) -> dict:
        """Return emission statistics."""
        return {
            "events_emitted": self._event_count,
            "errors": self._error_count,
            "output_file": str(self._output_file),
        }

    def load_events_from_jsonl(self, filepath: str) -> list[StoreEvent]:
        """
        Load and validate events from a JSONL file.
        
        Useful for re-ingesting previously emitted events.
        """
        events = []
        path = Path(filepath)
        
        if not path.exists():
            logger.warning(f"JSONL file not found: {filepath}")
            return events

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    event = StoreEvent(**data)
                    events.append(event)
                except (json.JSONDecodeError, ValidationError) as e:
                    self._error_count += 1
                    logger.error(
                        f"Line {line_num} in {filepath}: {e}"
                    )

        logger.info(
            f"Loaded {len(events)} events from {filepath} "
            f"({self._error_count} errors)"
        )
        return events
