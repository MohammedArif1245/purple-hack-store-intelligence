# Store Intelligence System

Turn raw CCTV footage into a live store analytics API that tells Apex Retail how many customers visited, where they went, and how many bought.

## Setup in 5 Commands (Local Development)

This approach uses SQLite and in-memory cache for zero-setup local testing.

```bash
# 1. Install API dependencies
pip install -r requirements.api.txt

# 2. Install Pipeline dependencies (Warning: Includes PyTorch, ~2GB)
pip install -r requirements.pipeline.txt

# 3. Install Dashboard dependencies
pip install -r requirements.dashboard.txt

# 4. Start the API server
python -m uvicorn app.main:app --reload --port 8000

# 5. Start the Live Dashboard (in a new terminal)
streamlit run dashboard/app.py
```

## Running via Docker (Production / Acceptance Gate)

To run the full stack (PostgreSQL + Redis + API + Dashboard) via Docker:

```bash
docker compose up --build
```
*Note: The pipeline script is run separately or via a dedicated worker container.*

## Running the Detection Pipeline

To process CCTV clips and send events to the running API:

```bash
# Process clips in the ./clips directory and send to localhost:8000
./pipeline/run.sh ./clips http://localhost:8000
```
*(If no clips are found, it will automatically send `sample_events.jsonl` to verify the API works).*

## API Endpoints (cURL Examples)

**Ingest Events**
```bash
curl -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d '{"events": [{"event_id": "test1", "store_id": "STORE_BLR_002", "visitor_id": "V1", "event_type": "ENTRY", "timestamp": "2026-03-03T10:00:00Z", "confidence": 0.9}]}'
```

**Get Live Metrics**
```bash
curl http://localhost:8000/stores/STORE_BLR_002/metrics
```

**Get Conversion Funnel**
```bash
curl http://localhost:8000/stores/STORE_BLR_002/funnel
```

**Get Zone Heatmap**
```bash
curl http://localhost:8000/stores/STORE_BLR_002/heatmap
```

**Check Anomalies**
```bash
curl http://localhost:8000/stores/STORE_BLR_002/anomalies
```

**System Health**
```bash
curl http://localhost:8000/health
```

## Testing

Run the full test suite with coverage:
```bash
pip install -r requirements.test.txt
pytest tests/ --cov=app --cov-report=term-missing
```
