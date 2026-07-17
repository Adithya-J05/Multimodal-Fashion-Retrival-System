"""FAISS vector query handling logic."""

import os
import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict, Any

from ..utils import get_logger
from ..indexer.extractors import FashionCLIPExtractor, DeviceManager

logger = get_logger(__name__)


class VectorSearchError(Exception):
    """Custom exception for vector search errors."""
    pass


class VectorSearcher:
    """
    Execute FAISS vector search operations with proper error handling.
    """
    
    def __init__(
        self,
        index_path: str,
        fashion_clip_extractor: Optional[FashionCLIPExtractor] = None,
        device_manager: Optional[DeviceManager] = None
    ):
        """
        Initialize vector searcher.
        
        Args:
            index_path: Path to FAISS index file
            fashion_clip_extractor: FashionCLIP extractor instance
            device_manager: Device manager instance
        """
        self.index_path = index_path
        self.device_manager = device_manager or DeviceManager()
        
        if fashion_clip_extractor is None:
            fashion_clip_extractor = FashionCLIPExtractor(
                device_manager=self.device_manager
            )
        self.fashion_clip = fashion_clip_extractor
        
        self._index = None
        self._id_mapping = []  # Store mapping from index position to image_id
        self._load_index()
        
    def _load_index(self) -> None:
        """Load FAISS index from disk with validation."""
        if not os.path.exists(self.index_path):
            raise VectorSearchError(
                f"FAISS index file not found: {self.index_path}. "
                "Please run indexing pipeline first."
            )
            
        try:
            self._index = faiss.read_index(self.index_path)
            # Initialize id mapping with sequential IDs (0 to n-1)
            # Since FAISS doesn't store IDs, we assume sequential mapping
            self._id_mapping = list(range(self._index.ntotal))
            logger.info(f"Loaded FAISS index from {self.index_path} with {self._index.ntotal} vectors")
            
        except Exception as e:
            raise VectorSearchError(
                f"Failed to load FAISS index from {self.index_path}: {str(e)}"
            )
            
    def encode_query(self, query_string: str) -> np.ndarray:
        """
        Encode query string using FashionCLIP text encoder.
        
        Args:
            query_string: Natural language query
            
        Returns:
            512-dimensional query embedding
        """
        try:
            embedding = self.fashion_clip.encode_text(query_string)
            return embedding
            
        except Exception as e:
            raise VectorSearchError(f"Failed to encode query: {str(e)}")
            
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute KNN search on FAISS index.
        
        Args:
            query_embedding: 512-d query embedding
            k: Number of candidates to retrieve
            
        Returns:
            Tuple of (distances, indices)
        """
        if self._index is None:
            raise VectorSearchError("FAISS index not loaded")
            
        if self._index.ntotal == 0:
            raise VectorSearchError("FAISS index is empty")
            
        k = min(k, self._index.ntotal)
        
        try:
            distances, indices = self._index.search(query_embedding.reshape(1, -1), k)
            return distances[0], indices[0]
            
        except Exception as e:
            raise VectorSearchError(f"Search failed: {str(e)}")
            
    def get_image_id_at_index(self, index_position: int) -> Optional[int]:
        """
        Get image_id at a specific index position.
        
        Args:
            index_position: Position in the index
            
        Returns:
            image_id or None if out of bounds
        """
        if 0 <= index_position < len(self._id_mapping):
            return self._id_mapping[index_position]
        return None
        
    def get_image_ids_by_indices(self, indices: np.ndarray) -> List[int]:
        """
        Get image_ids for a list of index positions.
        
        Args:
            indices: Array of index positions
            
        Returns:
            List of image_ids
        """
        if not hasattr(self, '_id_mapping') or not self._id_mapping:
            # If no mapping exists, assume indices are image_ids directly
            return [int(idx) for idx in indices if idx >= 0]
        
        return [self._id_mapping[idx] for idx in indices if 0 <= idx < len(self._id_mapping)]
            
    def get_index_size(self) -> int:
        """Get number of vectors in index."""
        return self._index.ntotal if self._index else 0
        
    def is_loaded(self) -> bool:
        """Check if index is loaded."""
        return self._index is not None