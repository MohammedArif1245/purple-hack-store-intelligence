"""
Store Intelligence — Staff Detection

Two-pass classifier to identify store employees:

Pass 1: Extract HSV histogram of person crop; compare against configured
        uniform colour ranges (narrow hue band, high saturation).
        If match probability > 0.7, flag is_staff=true.

Pass 2 (fallback): CLIP zero-shot with prompts
        ["a retail store employee in uniform", "a customer shopping"].
        Use softmax probability.

Staff events are stored raw — exclusion happens at query time in the API.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StaffDetector:
    """
    Two-pass staff classifier: HSV heuristic + optional CLIP fallback.
    
    Staff are identified but NOT removed from events — the API
    filters them at query time using is_staff=true.
    """

    def __init__(
        self,
        hue_range: tuple[int, int] = (100, 130),
        saturation_min: int = 80,
        value_min: int = 50,
        hsv_threshold: float = 0.7,
        use_clip_fallback: bool = False,
    ):
        """
        Args:
            hue_range: Expected hue range of staff uniform (OpenCV 0-180)
            saturation_min: Minimum saturation for uniform colour
            value_min: Minimum value for uniform colour
            hsv_threshold: Match probability threshold for HSV pass
            use_clip_fallback: Whether to use CLIP as fallback
        """
        self.hue_range = hue_range
        self.saturation_min = saturation_min
        self.value_min = value_min
        self.hsv_threshold = hsv_threshold
        self.use_clip_fallback = use_clip_fallback
        self._clip_model = None

    def classify(self, person_crop: np.ndarray) -> tuple[bool, float]:
        """
        Classify a person as staff or customer.
        
        Args:
            person_crop: BGR image of cropped person
            
        Returns:
            (is_staff: bool, confidence: float)
        """
        if person_crop is None or person_crop.size == 0:
            return False, 0.0

        # Pass 1: HSV colour heuristic
        is_staff, conf = self._hsv_classify(person_crop)
        if conf >= self.hsv_threshold:
            return is_staff, conf

        # Pass 2: CLIP zero-shot fallback (optional)
        if self.use_clip_fallback:
            return self._clip_classify(person_crop)

        return False, 1.0 - conf  # Default to customer

    def _hsv_classify(self, person_crop: np.ndarray) -> tuple[bool, float]:
        """
        HSV histogram-based uniform detection.
        
        Checks if the dominant colour of the person's clothing
        falls within the configured uniform colour range.
        """
        try:
            hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)

            # Focus on the torso region (middle 40% of height)
            h, w = hsv.shape[:2]
            torso = hsv[int(h * 0.2):int(h * 0.6), :, :]

            if torso.size == 0:
                return False, 0.0

            # Create mask for uniform colour range
            lower = np.array([
                self.hue_range[0], self.saturation_min, self.value_min
            ])
            upper = np.array([self.hue_range[1], 255, 255])
            mask = cv2.inRange(torso, lower, upper)

            # Calculate ratio of matching pixels
            match_ratio = np.sum(mask > 0) / mask.size

            return match_ratio >= self.hsv_threshold, float(match_ratio)

        except Exception as e:
            logger.error(f"HSV classification failed: {e}")
            return False, 0.0

    def _clip_classify(self, person_crop: np.ndarray) -> tuple[bool, float]:
        """
        CLIP zero-shot classification fallback.
        
        Uses text prompts to classify staff vs customer.
        Only loaded if use_clip_fallback=True and CLIP is available.
        """
        try:
            if self._clip_model is None:
                self._load_clip()

            if self._clip_model is None:
                return False, 0.0

            import torch
            from PIL import Image

            # Convert BGR to RGB PIL Image
            rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            # Encode image and text
            prompts = [
                "a retail store employee in uniform",
                "a customer shopping in a store"
            ]

            inputs = self._clip_processor(
                text=prompts,
                images=pil_image,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1).cpu().numpy()[0]

            # probs[0] = staff probability, probs[1] = customer probability
            is_staff = probs[0] > probs[1]
            confidence = float(max(probs))

            return bool(is_staff), confidence

        except Exception as e:
            logger.error(f"CLIP classification failed: {e}")
            return False, 0.0

    def _load_clip(self):
        """Lazy-load CLIP model."""
        try:
            from transformers import CLIPProcessor, CLIPModel

            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            logger.info("CLIP model loaded for staff detection fallback")
        except ImportError:
            logger.warning(
                "transformers not installed — CLIP fallback disabled"
            )
            self._clip_model = None
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            self._clip_model = None

    def update_uniform_config(
        self,
        hue_range: tuple[int, int],
        saturation_min: int = 80,
        value_min: int = 50,
    ):
        """Update uniform colour configuration (e.g., from store_layout.json)."""
        self.hue_range = hue_range
        self.saturation_min = saturation_min
        self.value_min = value_min
        logger.info(f"Updated uniform config: hue={hue_range}, sat>={saturation_min}")
