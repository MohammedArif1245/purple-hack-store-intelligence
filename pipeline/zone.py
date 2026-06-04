"""
Store Intelligence — Zone Classification

Loads store_layout.json and builds a {zone_id: Polygon} map using Shapely.
For each tracked person, computes bounding box centroid (cx, cy) and tests
polygon.contains(Point(cx, cy)).

Emits:
- ZONE_ENTER on first positive containment
- ZONE_EXIT on first negative after positive
- ZONE_DWELL every 30s of continuous containment
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


class ZoneClassifier:
    """
    Zone classification using polygon containment.
    
    Pure geometry: a bounding box centroid falls inside a polygon → 
    person is in that zone. No ML needed.
    """

    DWELL_INTERVAL_MS = 30000  # Emit ZONE_DWELL every 30 seconds

    def __init__(self, layout_path: str = "store_layout.json"):
        """
        Args:
            layout_path: Path to store_layout.json with zone polygon definitions
        """
        self.zones: dict[str, Polygon] = {}
        self.zone_types: dict[str, str] = {}
        self._load_layout(layout_path)

        # Track person-zone state: {visitor_id: {zone_id: {"enter_time": float, "last_dwell": float}}}
        self._person_zones: dict[str, dict[str, dict]] = {}

    def _load_layout(self, layout_path: str):
        """Load zone polygons from store_layout.json."""
        path = Path(layout_path)
        if not path.exists():
            logger.warning(f"Store layout not found: {layout_path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                layout = json.load(f)

            zones_data = layout.get("zones", {})
            for zone_id, zone_info in zones_data.items():
                polygon_coords = zone_info.get("polygon", [])
                if len(polygon_coords) >= 3:
                    self.zones[zone_id] = Polygon(polygon_coords)
                    self.zone_types[zone_id] = zone_info.get("type", "unknown")

            logger.info(f"Loaded {len(self.zones)} zones from {layout_path}")

        except Exception as e:
            logger.error(f"Failed to load store layout: {e}")

    def classify_position(
        self, cx: float, cy: float
    ) -> list[str]:
        """
        Determine which zones contain the given centroid.
        
        Args:
            cx, cy: Bounding box centroid coordinates
            
        Returns:
            List of zone_ids the point falls within
        """
        point = Point(cx, cy)
        return [
            zone_id for zone_id, polygon in self.zones.items()
            if polygon.contains(point)
        ]

    def update_person_zones(
        self,
        visitor_id: str,
        cx: float,
        cy: float,
        current_time: float,
    ) -> list[dict]:
        """
        Update zone tracking for a person and generate zone events.
        
        Args:
            visitor_id: Person identifier
            cx, cy: Current centroid position
            current_time: Current timestamp (epoch seconds)
            
        Returns:
            List of zone events to emit:
            [{"type": "ZONE_ENTER/EXIT/DWELL", "zone_id": str, "dwell_ms": int}]
        """
        current_zones = set(self.classify_position(cx, cy))
        events = []

        if visitor_id not in self._person_zones:
            self._person_zones[visitor_id] = {}

        prev_zones = set(self._person_zones[visitor_id].keys())

        # New zone entries
        entered = current_zones - prev_zones
        for zone_id in entered:
            self._person_zones[visitor_id][zone_id] = {
                "enter_time": current_time,
                "last_dwell": current_time,
            }
            events.append({
                "type": "ZONE_ENTER",
                "zone_id": zone_id,
                "dwell_ms": 0,
            })

        # Zone exits
        exited = prev_zones - current_zones
        for zone_id in exited:
            zone_state = self._person_zones[visitor_id].pop(zone_id, {})
            enter_time = zone_state.get("enter_time", current_time)
            dwell_ms = int((current_time - enter_time) * 1000)
            events.append({
                "type": "ZONE_EXIT",
                "zone_id": zone_id,
                "dwell_ms": dwell_ms,
            })

        # Dwell events for continuing zones (every 30s)
        for zone_id in current_zones & prev_zones:
            zone_state = self._person_zones[visitor_id][zone_id]
            time_since_last_dwell = (current_time - zone_state["last_dwell"]) * 1000

            if time_since_last_dwell >= self.DWELL_INTERVAL_MS:
                enter_time = zone_state["enter_time"]
                total_dwell_ms = int((current_time - enter_time) * 1000)
                events.append({
                    "type": "ZONE_DWELL",
                    "zone_id": zone_id,
                    "dwell_ms": total_dwell_ms,
                })
                zone_state["last_dwell"] = current_time

        return events

    def clear_person(self, visitor_id: str):
        """Remove zone tracking state for a person (on EXIT event)."""
        self._person_zones.pop(visitor_id, None)

    def is_billing_zone(self, zone_id: str) -> bool:
        """Check if a zone is a checkout/billing zone."""
        return self.zone_types.get(zone_id) == "checkout"

    def is_entry_zone(self, zone_id: str) -> bool:
        """Check if a zone is an entry/exit zone."""
        return self.zone_types.get(zone_id) == "entry_exit"
