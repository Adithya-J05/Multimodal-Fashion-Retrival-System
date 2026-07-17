"""Main orchestration driver for search workflows."""

import os
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ..utils import get_logger, load_json_file
from ..indexer.extractors import FashionCLIPExtractor, DeviceManager
from .search import VectorSearcher, VectorSearchError
from .reranker import HybridReranker

logger = get_logger(__name__)


class RetrievalPipeline:
    """
    Main retrieval pipeline orchestrating search and reranking.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        vector_searcher: Optional[VectorSearcher] = None,
        reranker: Optional[HybridReranker] = None
    ):
        """
        Initialize retrieval pipeline.
        
        Args:
            config: Configuration dictionary
            vector_searcher: Vector searcher instance
            reranker: Hybrid reranker instance
        """
        self.config = config
        self.device_manager = DeviceManager(
            preferred_device=config.get("device", {}).get("preferred", "auto"),
            fallback_to_cpu=config.get("device", {}).get("fallback_to_cpu", True)
        )
        
        paths = config.get("paths", {})
        self.db_path = paths.get("metadata_db", "data/processed/metadata.db")
        self.index_path = paths.get("index_file", "data/processed/vectors.index")
        
        # Validate files exist
        self._validate_files()
        
        # Initialize components
        self.fashion_clip = FashionCLIPExtractor(
            device_manager=self.device_manager
        )
        
        if vector_searcher is None:
            vector_searcher = VectorSearcher(
                index_path=self.index_path,
                fashion_clip_extractor=self.fashion_clip,
                device_manager=self.device_manager
            )
        self.vector_searcher = vector_searcher
        
        retrieval_config = config.get("retrieval", {})
        weights = retrieval_config.get("weights", {})
        
        if reranker is None:
            reranker = HybridReranker(
                image_weight=weights.get("image_similarity", 0.6),
                caption_weight=weights.get("caption_similarity", 0.3),
                attribute_weight=weights.get("attribute_match", 0.1)
            )
        self.reranker = reranker
        
        self.default_k = retrieval_config.get("default_k", 50)
        self.max_k = retrieval_config.get("max_k", 500)
        
        logger.info("Initialized RetrievalPipeline")
        
    def _validate_files(self) -> None:
        """Validate required files exist."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Please run indexing pipeline first."
            )
            
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"Metadata database not found at {self.db_path}. "
                "Please run indexing pipeline first."
            )
            
    def _fetch_metadata_batch(self, image_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Fetch metadata records for a list of image IDs.
        
        Args:
            image_ids: List of image IDs
            
        Returns:
            List of metadata records
        """
        if not image_ids:
            return []
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join(['?'] * len(image_ids))
        query = f"""
            SELECT image_id, file_path, caption, attributes,
                   image_embedding, caption_embedding,
                   fashionpedia_image_id, has_fashionpedia_annotations
            FROM metadata
            WHERE image_id IN ({placeholders})
        """
        
        cursor.execute(query, image_ids)
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            record = {
                "image_id": row[0],
                "file_path": row[1],
                "caption": row[2],
                "attributes": self._parse_attributes(row[3]),
                "image_embedding": self._parse_embedding(row[4]),
                "caption_embedding": self._parse_embedding(row[5]),
                "fashionpedia_image_id": row[6],
                "has_fashionpedia_annotations": bool(row[7])
            }
            records.append(record)
            
        # Maintain original order
        id_to_record = {r["image_id"]: r for r in records}
        ordered_records = [id_to_record.get(id) for id in image_ids if id in id_to_record]
        
        return ordered_records
        
    def _parse_attributes(self, attributes_json: str) -> Dict[str, Any]:
        """
        Parse attributes JSON safely.
        
        Args:
            attributes_json: JSON string of attributes
            
        Returns:
            Parsed attributes dictionary
        """
        if not attributes_json:
            return {
                "environment": "unknown",
                "clothing_items": [],
                "color_palette": [],
                "style_vibe": "unknown",
                "detected_objects": []
            }
            
        try:
            import json
            attrs = json.loads(attributes_json)
            
            # Ensure all required fields exist
            required_fields = {
                "environment": "unknown",
                "clothing_items": [],
                "color_palette": [],
                "style_vibe": "unknown",
                "detected_objects": []
            }
            
            for field, default in required_fields.items():
                if field not in attrs:
                    attrs[field] = default
                    
            return attrs
            
        except Exception as e:
            logger.warning(f"Failed to parse attributes: {e}")
            return {
                "environment": "unknown",
                "clothing_items": [],
                "color_palette": [],
                "style_vibe": "unknown",
                "detected_objects": []
            }
            
    def _parse_embedding(self, embedding_blob: bytes) -> Optional[np.ndarray]:
        """
        Parse embedding from blob safely.
        
        Args:
            embedding_blob: Binary embedding data
            
        Returns:
            Embedding as numpy array or None
        """
        if not embedding_blob:
            return None
            
        try:
            return np.frombuffer(embedding_blob, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to parse embedding: {e}")
            return None
            
    def search(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute full search pipeline.
        
        Args:
            query: Natural language query string
            k: Number of initial candidates to retrieve
            
        Returns:
            List of ranked result dictionaries
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
            
        k = k or self.default_k
        k = min(k, self.max_k)
        
        logger.info(f"Processing query: '{query}' with k={k}")
        
        try:
            # Step 1: Encode query
            query_embedding = self.vector_searcher.encode_query(query)
            
            # Step 2: Execute KNN search
            distances, indices = self.vector_searcher.search(query_embedding, k)
            
            if len(indices) == 0:
                logger.warning("No results found")
                return []
                
            # Step 3: Get image IDs from index positions
            image_ids = self.vector_searcher.get_image_ids_by_indices(indices)
            
            if not image_ids:
                logger.warning("Failed to map indices to image IDs")
                return []
                
            # Step 4: Fetch metadata
            metadata_records = self._fetch_metadata_batch(image_ids)
            
            if not metadata_records:
                logger.warning("No metadata records found")
                return []
                
            # Step 5: Rerank
            results = self.reranker.rerank(
                query_string=query,
                query_embedding=query_embedding,
                candidate_ids=image_ids,
                metadata_records=metadata_records,
                faiss_distances=distances.tolist() if distances is not None else None
            )
            
            logger.info(f"Returning {len(results)} results")
            return results
            
        except VectorSearchError as e:
            logger.error(f"Vector search failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Search pipeline failed: {e}")
            raise
            
    def search_batch(
        self,
        queries: List[str],
        k: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Execute batch search for multiple queries.
        
        Args:
            queries: List of natural language query strings
            k: Number of initial candidates to retrieve
            
        Returns:
            List of ranked result lists
        """
        results = []
        for query in queries:
            try:
                result = self.search(query, k)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed for query '{query}': {e}")
                results.append([])
                
        return results
        
    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get embedding for a query string.
        
        Args:
            query: Query string
            
        Returns:
            512-dimensional query embedding
        """
        return self.vector_searcher.encode_query(query)