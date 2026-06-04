# Store Intelligence System — Architecture & Design

## Architecture Overview
The Store Intelligence System transforms raw CCTV footage into actionable retail insights in real-time. It operates as a 4-stage pipeline:

1. **Detection Pipeline:** Raw video is processed frame-by-frame. `YOLOv8n` detects people with a lowered confidence floor of 0.30 to catch partial occlusions. `ByteTrack` provides multi-object tracking to assign unique track IDs within a clip. `OSNet Re-ID` creates 512-dimensional embeddings for each person to recognize them across different cameras and handle re-entries. A zone classifier uses Shapely geometry to map bounding box centroids to predefined store zones. A staff detector uses HSV color thresholds and an optional CLIP fallback to identify employees. Cross-camera deduplication merges duplicate detections from overlapping views.
2. **Event Stream:** The detection pipeline emits structured, schema-validated JSONL events (e.g., ENTRY, ZONE_ENTER, BILLING_QUEUE_JOIN). A POS correlator links these movement events with actual POS transactions using a 5-minute sliding window to confirm conversions.
3. **Intelligence API:** A robust FastAPI backend provides standard REST endpoints. Events are bulk-ingested asynchronously into a PostgreSQL 15 database (via SQLAlchemy async). A suite of endpoints (`/metrics`, `/funnel`, `/heatmap`, `/anomalies`, `/health`) queries this data to provide instant statistics. Redis 7 is intended for fast caching and queue-depth counting. The database design supports both local development (SQLite) and production (PostgreSQL) seamlessly.
4. **Live Dashboard:** A Streamlit web application polls the API every 2 seconds, displaying live KPIs (conversion rate, visitor counts), zone heatmaps, and alerting on anomalies like queue spikes or conversion drops.

## Storage Engine Rationale (PostgreSQL + Redis split)
We split storage between PostgreSQL and Redis to optimize for different access patterns. 
*   **PostgreSQL 15** is the persistent source of truth. It handles complex, relational analytics queries (like funnel drop-offs and session materialization) and bulk, idempotent inserts (via `ON CONFLICT`). Its strong ACID guarantees ensure no event data is lost.
*   **Redis 7** (or in-memory cache for local dev) is designed for transient, high-velocity data. It stores OSNet embeddings with short 30-minute TTLs for fast re-entry matching and tracks live counters like instantaneous queue depth. This prevents overwhelming the relational DB with high-frequency updates.

## Handling Confidence Degradation
Confidence degradation (e.g., due to occlusion or poor lighting) is handled systematically across the pipeline:
1.  **Detection Floor:** We lowered the YOLO confidence threshold from 0.5 to 0.3. This ensures partially occluded shoppers aren't silently dropped.
2.  **Tracking Resilience:** ByteTrack uses Kalman filters to predict positions when detections briefly fail, bridging gaps in tracking.
3.  **Data Preservation:** Every event logs the actual detection confidence. Downstream API endpoints and analytics can weigh or filter events based on these confidence scores, maintaining transparency rather than destroying data at the source.

## AI-Assisted Decisions
1.  **Database Dual-Mode Implementation:** An LLM suggested supporting SQLite for local development alongside PostgreSQL for Docker. I agreed and implemented this via an environment variable check. It drastically simplifies local testing without compromising the final Docker requirement.
2.  **OSNet Re-ID vs Simple Tracking:** I considered relying solely on ByteTrack across overlapping cameras. AI suggested integrating OSNet-x0.25 for true appearance-based Re-ID to solve cross-camera deduplication and re-entries. I agreed, as simple tracking loses identity when a person leaves a frame.
3.  **Idempotent Event Ingestion:** AI recommended using `ON CONFLICT DO NOTHING` for the POST `/ingest` endpoint to ensure batch retries don't duplicate data. I implemented this for robustness.
