# Architectural Choices & Trade-offs

## 1. Detection Model Selection
**Options Considered:**
*   YOLOv8n (Nano)
*   YOLOv8s/m (Small/Medium)
*   RT-DETR
*   MediaPipe

**AI Suggestion:** The AI recommended YOLOv8n because it provides the best balance of speed and accuracy on consumer hardware, particularly when running without dedicated, high-end GPUs. It explicitly noted that YOLOv8n's COCO pre-training already includes a robust 'person' class.

**My Choice & Why:** I chose **YOLOv8n**. Given the constraint of processing video feeds efficiently (potentially multiple feeds), speed is critical. While RT-DETR or YOLOv8m might offer marginally better bounding boxes, YOLOv8n runs fast enough to allow us to process every 3rd frame in near real-time, even on a CPU. Its native integration with ByteTrack in the `ultralytics` library also significantly simplified the pipeline architecture.

## 2. Event Schema Design
**Options Considered:**
*   A flat schema with all possible fields optional.
*   A strongly typed schema using Pydantic with strict event types and specific metadata validation.

**AI Suggestion:** The AI strongly recommended a strict, union-like Pydantic v2 schema (`StoreEvent`) where `event_type` is an Enum and required fields vary logically based on the event (e.g., `zone_id` required for `ZONE_ENTER`).

**My Choice & Why:** I implemented the **strict Pydantic v2 schema**. In an event-driven analytics system, data quality is paramount. By enforcing strict schemas at the `emit.py` boundary and the API ingestion boundary, we guarantee that bad data never corrupts the analytics database. Trade-off: It requires more boilerplate code, but it eliminates an entire class of runtime errors downstream (e.g., trying to calculate dwell time without a zone ID).

## 3. API Architecture (Sync vs Async & Storage)
**Options Considered:**
*   Synchronous FastAPI with standard SQLAlchemy and SQLite.
*   Fully Asynchronous FastAPI with asyncpg/aiosqlite and PostgreSQL/SQLite dual-mode.

**AI Suggestion:** The AI suggested a fully asynchronous architecture using `sqlalchemy[asyncio]` to handle concurrent batch ingestions efficiently without blocking the event loop.

**My Choice & Why:** I chose the **Fully Asynchronous Architecture with Dual-Mode Storage**. By using `async Session` and `asyncpg` (PostgreSQL) / `aiosqlite` (SQLite), the API can handle high-throughput event POSTs from multiple camera pipelines simultaneously. The dual-mode storage decision was critical: it provides a zero-setup local developer experience (SQLite) while satisfying the production acceptance gate of a Dockerized PostgreSQL instance. Trade-off: Async SQLAlchemy is more complex to debug and configure, but the performance benefits for batch ingestion at scale are necessary for a retail CCTV system.
