"""
Store Intelligence System — Pydantic v2 Models

All event schemas, API request/response models, and data validation.
Every component validates against these models.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Event Types ─────────────────────────────────────────────────────────────

class EventType(str, Enum):
    """All 8 event types emitted by the detection pipeline."""
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"


class AnomalyType(str, Enum):
    """Detectable anomaly categories."""
    BILLING_QUEUE_SPIKE = "BILLING_QUEUE_SPIKE"
    CONVERSION_DROP = "CONVERSION_DROP"
    DEAD_ZONE = "DEAD_ZONE"


class Severity(str, Enum):
    """Anomaly severity levels."""
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


class ServiceStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


# ─── Event Metadata ──────────────────────────────────────────────────────────

class EventMetadata(BaseModel):
    """Optional metadata attached to events."""
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: Optional[int] = None
    reid_confidence: Optional[float] = None
    embedding_hash: Optional[str] = None


# ─── Core Event Schema ───────────────────────────────────────────────────────

class StoreEvent(BaseModel):
    """
    Core event schema — every detection pipeline output validates against this.
    
    Schema covers all 8 event types: ENTRY, EXIT, ZONE_ENTER, ZONE_EXIT,
    ZONE_DWELL, BILLING_QUEUE_JOIN, BILLING_QUEUE_ABANDON, REENTRY.
    """
    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUIDv4 unique event identifier"
    )
    store_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Store identifier (e.g., STORE_BLR_002)"
    )
    camera_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Camera identifier (e.g., CAM_ENTRY_01)"
    )
    visitor_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique visitor identifier (e.g., VIS_c8a2f1)"
    )
    event_type: EventType = Field(
        ...,
        description="One of 8 valid event types"
    )
    timestamp: datetime = Field(
        ...,
        description="Event timestamp in UTC (ISO-8601)"
    )
    zone_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Zone name from store_layout.json"
    )
    dwell_ms: int = Field(
        default=0,
        ge=0,
        description="Dwell time in milliseconds"
    )
    is_staff: bool = Field(
        default=False,
        description="True if person is detected as staff"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score (0.0 to 1.0)"
    )
    metadata: Optional[EventMetadata] = Field(
        default=None,
        description="Optional event metadata"
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v):
        """Ensure all timestamps are UTC."""
        if isinstance(v, str):
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        elif isinstance(v, datetime):
            dt = v
        else:
            raise ValueError(f"Invalid timestamp format: {v}")
        
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    @model_validator(mode="after")
    def validate_zone_for_zone_events(self):
        """Zone events must have a zone_id."""
        zone_events = {
            EventType.ZONE_ENTER, EventType.ZONE_EXIT, 
            EventType.ZONE_DWELL, EventType.BILLING_QUEUE_JOIN,
            EventType.BILLING_QUEUE_ABANDON
        }
        if self.event_type in zone_events and not self.zone_id:
            raise ValueError(
                f"zone_id is required for event type {self.event_type}"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "event_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "store_id": "STORE_BLR_002",
                    "camera_id": "CAM_ENTRY_01",
                    "visitor_id": "VIS_c8a2f1",
                    "event_type": "ZONE_DWELL",
                    "timestamp": "2026-03-03T14:22:10Z",
                    "zone_id": "SKINCARE",
                    "dwell_ms": 8400,
                    "is_staff": False,
                    "confidence": 0.91,
                    "metadata": {
                        "queue_depth": None,
                        "sku_zone": "MOISTURISER",
                        "session_seq": 5
                    }
                }
            ]
        }
    }


# ─── API Request Models ──────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Batch event ingestion request. Max 500 events per batch."""
    events: list[StoreEvent] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Batch of events to ingest (max 500)"
    )


# ─── API Response Models ─────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response from /events/ingest endpoint."""
    success_count: int = Field(
        ..., ge=0,
        description="Number of events successfully ingested"
    )
    failed_count: int = Field(
        ..., ge=0,
        description="Number of events that failed validation"
    )
    failed_ids: list[str] = Field(
        default_factory=list,
        description="Event IDs that failed ingestion"
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages for failed events"
    )


class ZoneDwell(BaseModel):
    """Dwell time statistics for a single zone."""
    zone_id: str
    avg_dwell_ms: float = 0.0
    visit_count: int = 0


class MetricsResponse(BaseModel):
    """Response from /stores/{id}/metrics endpoint."""
    store_id: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_visitors: int = 0
    conversion_rate: float = 0.0
    avg_dwell_ms: float = 0.0
    dwell_per_zone: list[ZoneDwell] = Field(default_factory=list)
    queue_depth: int = 0
    abandonment_rate: float = 0.0


class FunnelStage(BaseModel):
    """A single stage in the conversion funnel."""
    stage: str
    count: int = 0
    drop_off_pct: float = 0.0


class FunnelResponse(BaseModel):
    """Response from /stores/{id}/funnel endpoint."""
    store_id: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    stages: list[FunnelStage] = Field(default_factory=list)
    total_sessions: int = 0


class HeatmapZone(BaseModel):
    """Heatmap data for a single zone."""
    zone_id: str
    visit_freq: int = 0
    avg_dwell_ms: float = 0.0
    score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Normalised score 0-100"
    )
    data_confidence: str = Field(
        default="HIGH",
        description="HIGH if >= 20 sessions, LOW if < 20"
    )


class HeatmapResponse(BaseModel):
    """Response from /stores/{id}/heatmap endpoint."""
    store_id: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    zones: list[HeatmapZone] = Field(default_factory=list)


class AnomalyDetail(BaseModel):
    """A single detected anomaly."""
    anomaly_type: AnomalyType
    severity: Severity
    description: str = ""
    suggested_action: str = ""
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    details: dict = Field(default_factory=dict)


class AnomalyResponse(BaseModel):
    """Response from /stores/{id}/anomalies endpoint."""
    store_id: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    anomalies: list[AnomalyDetail] = Field(default_factory=list)
    anomaly_count: int = 0


class StoreHealth(BaseModel):
    """Health info for a single store's feed."""
    store_id: str
    last_event_at: Optional[datetime] = None
    is_stale: bool = False
    event_count_today: int = 0


class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    status: ServiceStatus = ServiceStatus.HEALTHY
    database: str = "connected"
    redis: dict = Field(default_factory=lambda: {"status": "not_configured", "note": "Redis not yet integrated"})
    uptime_seconds: float = 0.0
    stores: list[StoreHealth] = Field(default_factory=list)
    stale_feeds: list[str] = Field(
        default_factory=list,
        description="Store IDs with >10 min feed lag"
    )
