#!/bin/bash
# Store Intelligence — Pipeline Runner
# Processes all CCTV clips and sends events to the API.
#
# Usage:
#   ./pipeline/run.sh [clips_dir] [api_url]
#
# Arguments:
#   clips_dir  - Directory containing CCTV clips (default: ./clips)
#   api_url    - API base URL (default: http://localhost:8000)

set -euo pipefail

CLIPS_DIR="${1:-./clips}"
API_URL="${2:-http://localhost:8000}"
STORE_ID="STORE_BLR_002"
OUTPUT_DIR="./output"

echo "==================================================="
echo "  Store Intelligence — Detection Pipeline"
echo "==================================================="
echo "Clips directory: $CLIPS_DIR"
echo "API URL:         $API_URL"
echo "Store ID:        $STORE_ID"
echo "Output:          $OUTPUT_DIR"
echo "==================================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if clips directory exists
if [ ! -d "$CLIPS_DIR" ]; then
    echo "WARNING: Clips directory not found: $CLIPS_DIR"
    echo "Creating directory and using sample events instead..."
    mkdir -p "$CLIPS_DIR"
    
    # Use sample events for demo
    if [ -f "sample_events.jsonl" ]; then
        echo "Sending sample events to API..."
        python -c "
import json
import requests

events = []
with open('sample_events.jsonl', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            events.append(json.loads(line))

# Send in batches of 500
batch_size = 500
for i in range(0, len(events), batch_size):
    batch = events[i:i+batch_size]
    response = requests.post(
        '${API_URL}/events/ingest',
        json={'events': batch}
    )
    print(f'Batch {i//batch_size + 1}: {response.json()}')

print(f'Total events sent: {len(events)}')
"
        echo "Sample events ingested successfully!"
    fi
    exit 0
fi

# Process each video clip
echo ""
echo "Processing video clips..."
for clip in "$CLIPS_DIR"/*.mp4 "$CLIPS_DIR"/*.avi "$CLIPS_DIR"/*.mkv; do
    [ -f "$clip" ] || continue
    echo ""
    echo "Processing: $(basename $clip)"
    
python -c "
import sys
import time
from datetime import datetime, timezone
sys.path.insert(0, '.')
from pipeline.tracker import PersonTracker
from pipeline.emit import EventEmitter
from pipeline.zone import ZoneClassifier
from pipeline.staff import StaffDetector
from pipeline.reid import ReIDManager
from pipeline.dedup import Deduplicator

# Initialise components
tracker = PersonTracker()
emitter = EventEmitter(output_dir='$OUTPUT_DIR', store_id='$STORE_ID')
zoner = ZoneClassifier('store_layout.json')
staff_det = StaffDetector()
reid = ReIDManager(use_gpu=False)
dedup = Deduplicator()

track_state = {}

print(f'  Tracking persons in: $clip')
for frame_num, tracked, exited, frame in tracker.process_video('$clip', frame_skip=3):
    current_time = time.time()
    
    for person in tracked:
        track_id = person['track_id']
        bbox = person['bbox']
        cx, cy = person['centroid']
        
        if person['is_new']:
            crop = tracker.crop_person(frame, bbox)
            is_staff, staff_conf = staff_det.classify(crop)
            emb = reid.extract_embedding(crop)
            
            reentry_id = None
            if emb is not None:
                reentry_id = reid.check_reentry('$STORE_ID', emb)
                
            if reentry_id:
                vis_id = reentry_id
                is_reentry = True
            else:
                vis_id = f\"VIS_{track_id:04d}\"
                is_reentry = False
                
            vis_id = dedup.check_and_register('$STORE_ID', 'CAM_01', vis_id, emb, current_time)
            
            track_state[track_id] = {
                'visitor_id': vis_id,
                'is_staff': is_staff,
                'embedding': emb
            }
            
            if is_reentry:
                emitter.emit_reentry(
                    visitor_id=vis_id,
                    timestamp=datetime.now(timezone.utc),
                    camera_id='CAM_01',
                    confidence=person['confidence'],
                    reid_confidence=0.9
                )
            else:
                emitter.emit_entry(
                    visitor_id=vis_id,
                    timestamp=datetime.now(timezone.utc),
                    camera_id='CAM_01',
                    confidence=person['confidence'],
                    is_staff=is_staff
                )
                
        if track_id not in track_state:
            continue
            
        vis_id = track_state[track_id]['visitor_id']
        
        zone_events = zoner.update_person_zones(vis_id, cx, cy, current_time)
        for z_event in zone_events:
            if z_event['type'] == 'ZONE_ENTER':
                emitter.emit_zone_enter(vis_id, datetime.now(timezone.utc), 'CAM_01', z_event['zone_id'], person['confidence'])
                if zoner.is_billing_zone(z_event['zone_id']):
                    emitter.emit_billing_queue_join(vis_id, datetime.now(timezone.utc), 'CAM_01', person['confidence'], queue_depth=1)
            elif z_event['type'] == 'ZONE_EXIT':
                emitter.emit_zone_exit(vis_id, datetime.now(timezone.utc), 'CAM_01', z_event['zone_id'], z_event['dwell_ms'], person['confidence'])
                if zoner.is_billing_zone(z_event['zone_id']):
                    emitter.emit_billing_queue_abandon(vis_id, datetime.now(timezone.utc), 'CAM_01', person['confidence'], z_event['dwell_ms'], queue_depth=0)
            elif z_event['type'] == 'ZONE_DWELL':
                emitter.emit_zone_dwell(vis_id, datetime.now(timezone.utc), 'CAM_01', z_event['zone_id'], z_event['dwell_ms'], person['confidence'])

    for track_id in exited:
        if track_id in track_state:
            vis_id = track_state[track_id]['visitor_id']
            emb = track_state[track_id]['embedding']
            
            if emb is not None:
                reid.store_exit_embedding('$STORE_ID', str(track_id), vis_id, emb)
                
            zoner.clear_person(vis_id)
            emitter.emit_exit(
                visitor_id=vis_id,
                timestamp=datetime.now(timezone.utc),
                camera_id='CAM_01',
                confidence=0.8,
            )
            del track_state[track_id]

print(f'  Stats: {emitter.stats}')
"
done

# Send events to API
echo ""
echo "Sending events to API at $API_URL..."
EVENTS_FILE="$OUTPUT_DIR/events_${STORE_ID}.jsonl"

if [ -f "$EVENTS_FILE" ]; then
    python -c "
import json
import requests

events = []
with open('$EVENTS_FILE', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            events.append(json.loads(line))

batch_size = 500
for i in range(0, len(events), batch_size):
    batch = events[i:i+batch_size]
    response = requests.post(
        '${API_URL}/events/ingest',
        json={'events': batch}
    )
    result = response.json()
    print(f'Batch {i//batch_size + 1}: {result[\"success_count\"]} success, {result[\"failed_count\"]} failed')

print(f'Total events sent: {len(events)}')
"
    echo "Events ingested successfully!"
else
    echo "No events file found at $EVENTS_FILE"
fi

echo ""
echo "==================================================="
echo "  Pipeline complete!"
echo "  View metrics: curl $API_URL/stores/$STORE_ID/metrics"
echo "==================================================="
