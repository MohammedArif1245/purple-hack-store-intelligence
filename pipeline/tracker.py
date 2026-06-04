"""
Store Intelligence — ByteTrack Multi-Object Tracker

ByteTrack wrapper around ultralytics built-in tracker.
Assigns stable track_id per person within a clip.
Handles occlusion via Kalman filter prediction.

Critical tuning: iou_threshold=0.3 (lower than default) to prevent
merging bounding boxes of people walking close together (group entry case).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PersonTracker:
    """
    ByteTrack-based multi-object tracker for person tracking.
    
    Wraps ultralytics built-in tracker to assign and maintain
    unique track_ids across frames within a single video clip.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.30,
        iou_threshold: float = 0.3,
        tracker_type: str = "bytetrack.yaml",
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: YOLOv8 model variant
            confidence_threshold: Min confidence for detection (0.30 per spec)
            iou_threshold: IoU threshold for tracking association.
                          Set to 0.3 (lower than default) to prevent merging
                          bounding boxes of people walking close together.
            tracker_type: Tracker config file (bytetrack.yaml)
            device: 'cuda', 'cpu', or None for auto
        """
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.tracker_type = tracker_type
        self.device = device or ("cuda" if self._cuda_available() else "cpu")

        # Track state: {track_id: {...}} for maintaining history
        self._active_tracks: dict[int, dict] = {}
        self._exited_tracks: set[int] = set()
        self._frame_count = 0

        logger.info(
            f"PersonTracker initialized: tracker={tracker_type}, "
            f"iou={iou_threshold}, conf={confidence_threshold}, "
            f"device={self.device}"
        )

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def track_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection + tracking on a single frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of tracked persons, each with:
            - track_id: int unique within this clip
            - bbox: [x1, y1, x2, y2]
            - confidence: float
            - centroid: (cx, cy)
            - is_new: bool (first appearance)
            - frames_tracked: int
        """
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[0],  # person class only
            tracker=self.tracker_type,
            device=self.device,
            persist=True,
            verbose=False,
        )

        self._frame_count += 1
        current_track_ids = set()
        tracked_persons = []

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            for box, track_id_tensor in zip(result.boxes, result.boxes.id):
                track_id = int(track_id_tensor.cpu().numpy())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                current_track_ids.add(track_id)

                is_new = track_id not in self._active_tracks
                if is_new:
                    self._active_tracks[track_id] = {
                        "first_frame": self._frame_count,
                        "frames_seen": 0,
                        "last_centroid": None,
                    }

                track_info = self._active_tracks[track_id]
                track_info["frames_seen"] += 1
                track_info["last_centroid"] = (float(cx), float(cy))

                tracked_persons.append({
                    "track_id": track_id,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "centroid": (float(cx), float(cy)),
                    "is_new": is_new,
                    "frames_tracked": track_info["frames_seen"],
                })

        # Detect exits: tracks that were active but not in current frame
        disappeared = set(self._active_tracks.keys()) - current_track_ids - self._exited_tracks
        for track_id in disappeared:
            # Track might just be temporarily occluded
            # Only mark as exited after missing for several frames
            track_info = self._active_tracks[track_id]
            if "missing_frames" not in track_info:
                track_info["missing_frames"] = 0
            track_info["missing_frames"] += 1

        return tracked_persons

    def get_exited_tracks(self, missing_threshold: int = 30) -> list[int]:
        """
        Get track IDs that have been missing for more than threshold frames.
        These are likely exits.
        
        Args:
            missing_threshold: Frames a track must be missing to count as exit
            
        Returns:
            List of track_ids that exited
        """
        exited = []
        for track_id, info in list(self._active_tracks.items()):
            missing = info.get("missing_frames", 0)
            if missing >= missing_threshold and track_id not in self._exited_tracks:
                exited.append(track_id)
                self._exited_tracks.add(track_id)
        return exited

    def process_video(self, video_path: str, frame_skip: int = 3, max_frames: Optional[int] = None):
        """
        Process a video file with tracking.
        
        Args:
            video_path: Path to video file
            frame_skip: Process every Nth frame
            max_frames: Optional frame limit
            
        Yields:
            (frame_number, tracked_persons, exited_track_ids) per processed frame
        """
        path = Path(video_path)
        if not path.exists():
            logger.error(f"Video not found: {video_path}")
            return

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Tracking {video_path}: {total} frames @ {fps:.1f} FPS")

        frame_num = 0
        processed = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and processed >= max_frames:
                    break

                if frame_num % frame_skip != 0:
                    frame_num += 1
                    continue

                tracked = self.track_frame(frame)
                exited = self.get_exited_tracks()

                yield frame_num, tracked, exited, frame

                processed += 1
                frame_num += 1

        finally:
            cap.release()

    def reset(self):
        """Reset tracker state for a new video clip."""
        self._active_tracks.clear()
        self._exited_tracks.clear()
        self._frame_count = 0

    @property
    def active_count(self) -> int:
        """Number of currently active tracks."""
        return len(self._active_tracks) - len(self._exited_tracks)

    def crop_person(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        """
        Crop a person from the frame using their bounding box.
        Useful for Re-ID embedding extraction.
        
        Args:
            frame: Full BGR frame
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Cropped BGR image of the person
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]
