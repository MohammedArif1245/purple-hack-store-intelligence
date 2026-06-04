"""
Store Intelligence — OSNet Re-Identification

OSNet-x0.25 loaded via torchreid. Extracts 512-dimensional appearance
embeddings from cropped person bounding boxes.

On EXIT: stores embedding in cache with key reid:{store_id}:{track_id} (30-min TTL).
On new ENTRY: compares embedding against recent exits via cosine similarity.
Threshold >= 0.85 = same person → REENTRY event, reuse visitor_id.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ReIDManager:
    """
    Person re-identification using OSNet appearance embeddings.
    
    Handles re-entry detection (same person leaving and returning)
    and cross-camera identity matching.
    """

    SIMILARITY_THRESHOLD = 0.85
    EMBEDDING_DIM = 512
    CACHE_TTL_SECONDS = 1800  # 30 minutes

    def __init__(self, model_name: str = "osnet_x0_25", use_gpu: bool = False):
        """
        Args:
            model_name: torchreid model name (osnet_x0_25 pretrained on Market-1501)
            use_gpu: Whether to use GPU for inference
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self._model = None
        self._extractor = None

        # In-memory embedding store (replaces Redis for local mode)
        # {store_id: {track_id: {"embedding": np.array, "timestamp": float, "visitor_id": str}}}
        self._embedding_store: dict[str, dict[str, dict]] = {}

        logger.info(f"ReIDManager initialized: model={model_name}, gpu={use_gpu}")

    def _load_model(self):
        """Lazy-load the OSNet model."""
        if self._extractor is not None:
            return

        try:
            try:
                from torchreid.utils import FeatureExtractor
            except ImportError:
                from torchreid.reid.utils import FeatureExtractor

            self._extractor = FeatureExtractor(
                model_name=self.model_name,
                model_path="",  # Uses pretrained weights
                device="cuda" if self.use_gpu else "cpu",
            )
            logger.info(f"OSNet model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load OSNet model: {e}")
            self._extractor = None

    def extract_embedding(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 512-dim appearance embedding from a cropped person image.
        
        Args:
            person_crop: BGR image of a person (cropped from frame)
            
        Returns:
            512-dimensional numpy embedding, or None on failure
        """
        self._load_model()
        if self._extractor is None:
            return None

        try:
            import cv2
            # Resize to model input size
            resized = cv2.resize(person_crop, (128, 256))
            
            features = self._extractor(resized)
            embedding = features.cpu().numpy().flatten()

            # Normalize to unit vector for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2))

    def store_exit_embedding(
        self, store_id: str, track_id: str,
        visitor_id: str, embedding: np.ndarray
    ):
        """
        Store embedding on EXIT event for future re-entry matching.
        
        Args:
            store_id: Store identifier
            track_id: Track ID from ByteTrack
            visitor_id: Assigned visitor ID
            embedding: 512-dim appearance embedding
        """
        if store_id not in self._embedding_store:
            self._embedding_store[store_id] = {}

        self._embedding_store[store_id][track_id] = {
            "embedding": embedding,
            "timestamp": time.time(),
            "visitor_id": visitor_id,
        }

        # Clean up expired entries
        self._cleanup_expired(store_id)

        logger.debug(
            f"Stored exit embedding: store={store_id}, "
            f"track={track_id}, visitor={visitor_id}"
        )

    def check_reentry(
        self, store_id: str, embedding: np.ndarray
    ) -> Optional[str]:
        """
        Check if a new entry matches any recent exit.
        
        Args:
            store_id: Store identifier
            embedding: 512-dim embedding of the new entrant
            
        Returns:
            visitor_id of matching exit, or None if no match
        """
        if store_id not in self._embedding_store:
            return None

        self._cleanup_expired(store_id)

        best_match = None
        best_similarity = 0.0

        for track_id, data in self._embedding_store[store_id].items():
            similarity = self.cosine_similarity(embedding, data["embedding"])
            if similarity >= self.SIMILARITY_THRESHOLD and similarity > best_similarity:
                best_similarity = similarity
                best_match = data["visitor_id"]

        if best_match:
            logger.info(
                f"Re-entry detected: store={store_id}, "
                f"visitor={best_match}, similarity={best_similarity:.3f}"
            )

        return best_match

    def _cleanup_expired(self, store_id: str):
        """Remove embeddings older than TTL."""
        if store_id not in self._embedding_store:
            return

        now = time.time()
        expired = [
            tid for tid, data in self._embedding_store[store_id].items()
            if now - data["timestamp"] > self.CACHE_TTL_SECONDS
        ]
        for tid in expired:
            del self._embedding_store[store_id][tid]
