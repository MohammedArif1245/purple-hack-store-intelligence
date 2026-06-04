"""
Store Intelligence — YOLOv8n Person Detection

Entry point for video processing. Loads YOLOv8n with ultralytics.
Iterates frames at configured FPS (default: every 3rd frame for speed).
Filters detections to class == 0 (person). Confidence floor of 0.30
(not default 0.5) — low-confidence detections must be emitted, not silently dropped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PersonDetector:
    """
    YOLOv8n-based person detector.
    
    Uses pretrained COCO weights — no custom training needed.
    Person class is class_id=0 in COCO.
    """

    # COCO class ID for 'person'
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.30,
        frame_skip: int = 3,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: YOLOv8 model variant (auto-downloads if missing)
            confidence_threshold: Minimum confidence to keep a detection.
                                  Set to 0.30 (not default 0.5) to avoid
                                  silently dropping low-confidence detections.
            frame_skip: Process every Nth frame for speed
            device: 'cuda', 'cpu', or None for auto-detection
        """
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.frame_skip = frame_skip
        self.device = device or ("cuda" if self._cuda_available() else "cpu")

        logger.info(
            f"PersonDetector initialized: model={model_name}, "
            f"conf={confidence_threshold}, skip={frame_skip}, "
            f"device={self.device}"
        )

    @staticmethod
    def _cuda_available() -> bool:
        """Check if CUDA GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def detect_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Run person detection on a single frame.
        
        Args:
            frame: BGR image as numpy array (from OpenCV)
            
        Returns:
            List of detections, each with:
            - bbox: [x1, y1, x2, y2] coordinates
            - confidence: float detection confidence
            - class_id: int (always 0 for person)
            - centroid: (cx, cy) center point of bbox
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            classes=[self.PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                # Calculate centroid for zone classification
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class_id": cls_id,
                    "centroid": (float(cx), float(cy)),
                })

        return detections

    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> Generator[tuple[int, list[dict]], None, None]:
        """
        Process a video file, yielding detections per frame.
        
        Args:
            video_path: Path to the video file
            max_frames: Optional limit on total frames to process
            
        Yields:
            (frame_number, detections) for each processed frame
        """
        path = Path(video_path)
        if not path.exists():
            logger.error(f"Video file not found: {video_path}")
            return

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            f"Processing {video_path}: {total_frames} frames @ {fps:.1f} FPS, "
            f"processing every {self.frame_skip} frame(s)"
        )

        frame_num = 0
        processed = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and processed >= max_frames:
                    break

                # Skip frames for performance
                if frame_num % self.frame_skip != 0:
                    frame_num += 1
                    continue

                detections = self.detect_frame(frame)
                yield frame_num, detections

                processed += 1
                frame_num += 1

                if processed % 100 == 0:
                    logger.info(
                        f"Processed {processed} frames, "
                        f"frame {frame_num}/{total_frames}"
                    )

        finally:
            cap.release()

        logger.info(
            f"Finished processing {video_path}: "
            f"{processed} frames processed, {frame_num} total"
        )

    def get_video_info(self, video_path: str) -> dict:
        """Get metadata about a video file."""
        cap = cv2.VideoCapture(video_path)
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        info["duration_seconds"] = (
            info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
        )
        cap.release()
        return info
