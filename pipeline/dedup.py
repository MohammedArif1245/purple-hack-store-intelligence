"""
Store Intelligence — Cross-Camera Deduplication

Runs after tracking, before emit. Maintains a 60-second sliding window
of active embeddings per store. When two camera feeds produce detections
with cosine similarity > 0.85 within the same window, the later detection
is merged into the earlier visitor_id.

Prevents double-counting at entry/floor camera overlap.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Deduplicator:
    """
    Cross-camera deduplication using appearance embeddings.
    
    Maintains a sliding window of active embeddings and merges
    duplicate detections from overlapping camera views.
    """

    SIMILARITY_THRESHOLD = 0.85
    WINDOW_SECONDS = 60  # 60-second sliding window

    def __init__(self):
        # {store_id: [{embedding, visitor_id, camera_id, timestamp}]}
        self._window: dict[str, list[dict]] = {}
        self._merge_count = 0

    def check_and_register(
        self,
        store_id: str,
        camera_id: str,
        visitor_id: str,
        embedding: Optional[np.ndarray],
        timestamp: Optional[float] = None,
    ) -> str:
        """
        Check if this detection is a duplicate from another camera.
        
        If a match is found (cosine similarity > 0.85), returns the
        existing visitor_id (the later detection is merged).
        Otherwise, registers this as a new detection.
        
        Args:
            store_id: Store identifier
            camera_id: Source camera
            visitor_id: Proposed visitor ID for this detection
            embedding: 512-dim appearance embedding (None = skip dedup)
            timestamp: Detection time (defaults to now)
            
        Returns:
            Final visitor_id to use (may be different if merged)
        """
        if embedding is None:
            return visitor_id

        ts = timestamp or time.time()

        if store_id not in self._window:
            self._window[store_id] = []

        # Clean expired entries
        self._cleanup(store_id, ts)

        # Check against existing entries from OTHER cameras
        for entry in self._window[store_id]:
            if entry["camera_id"] == camera_id:
                continue  # Same camera — not a cross-cam duplicate

            similarity = float(np.dot(embedding, entry["embedding"]))
            if similarity >= self.SIMILARITY_THRESHOLD:
                self._merge_count += 1
                logger.info(
                    f"Dedup merge: {visitor_id} → {entry['visitor_id']} "
                    f"(cam {camera_id} ↔ {entry['camera_id']}, "
                    f"sim={similarity:.3f})"
                )
                return entry["visitor_id"]  # Use the earlier visitor_id

        # No duplicate found — register new entry
        self._window[store_id].append({
            "embedding": embedding,
            "visitor_id": visitor_id,
            "camera_id": camera_id,
            "timestamp": ts,
        })

        return visitor_id

    def _cleanup(self, store_id: str, current_time: float):
        """Remove entries outside the sliding window."""
        if store_id not in self._window:
            return

        cutoff = current_time - self.WINDOW_SECONDS
        self._window[store_id] = [
            e for e in self._window[store_id]
            if e["timestamp"] >= cutoff
        ]

    @property
    def merge_count(self) -> int:
        """Total number of duplicates merged."""
        return self._merge_count

    def reset(self):
        """Clear all deduplication state."""
        self._window.clear()
        self._merge_count = 0
