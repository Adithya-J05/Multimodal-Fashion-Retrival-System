"""Implementation of the hybrid weighted scoring formula."""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict

from ..utils import (
    get_logger,
    cosine_similarity,
    normalize_string,
    safe_divide,
    normalize_to_range
)

logger = get_logger(__name__)


class AttributeMatcher:
    """
    Handle attribute matching between query tokens and stored metadata.
    """
    
    def __init__(self, stop_words: Optional[Set[str]] = None):
        """
        Initialize attribute matcher.
        
        Args:
            stop_words: Set of stop words to ignore
        """
        self.stop_words = stop_words or {
            'a', 'an', 'the', 'and', 'or', 'but', 'for', 'nor', 'on', 'at', 
            'to', 'by', 'in', 'with', 'without', 'of', 'for', 'about', 'than',
            'like', 'so', 'too', 'very', 'just', 'some', 'any', 'more', 'most'
        }
        
    def extract_keywords(self, text: str) -> Set[str]:
        """
        Extract meaningful keywords from text.
        
        Args:
            text: Input text string
            
        Returns:
            Set of lowercase keywords
        """
        if not text:
            return set()
            
        # Normalize and split
        text = normalize_string(text)
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        
        # Filter stop words and short words
        keywords = {
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        }
        
        return keywords
        
    def calculate_attribute_match(
        self,
        query: str,
        metadata: Dict[str, Any],
        weight_threshold: float = 0.3
    ) -> float:
        """
        Calculate attribute match score between query and metadata.
        
        Args:
            query: User query string
            metadata: Stored metadata dictionary
            weight_threshold: Minimum weight for considering a match
            
        Returns:
            Attribute match score in range [0, 1]
        """
        if not query or not metadata:
            return 0.0
            
        query_keywords = self.extract_keywords(query)
        
        if not query_keywords:
            return 0.0
            
        # Collect all metadata text fields for matching
        metadata_text = []
        
        # Extract clothing items
        clothing_items = metadata.get("clothing_items", [])
        if isinstance(clothing_items, list):
            metadata_text.extend([str(item) for item in clothing_items])
            
        # Extract color palette
        color_palette = metadata.get("color_palette", [])
        if isinstance(color_palette, list):
            metadata_text.extend([str(color) for color in color_palette])
            
        # Extract environment
        environment = metadata.get("environment", "")
        if environment:
            metadata_text.append(str(environment))
            
        # Extract style vibe
        style_vibe = metadata.get("style_vibe", "")
        if style_vibe:
            metadata_text.append(str(style_vibe))
            
        # Extract detected objects
        detected_objects = metadata.get("detected_objects", [])
        if isinstance(detected_objects, list):
            metadata_text.extend([str(obj) for obj in detected_objects])
            
        if not metadata_text:
            return 0.0
            
        # Normalize and tokenize metadata
        metadata_keywords = set()
        for text in metadata_text:
            metadata_keywords.update(self.extract_keywords(text))
            
        if not metadata_keywords:
            return 0.0
            
        # Calculate overlap
        matches = query_keywords.intersection(metadata_keywords)
        
        if not matches:
            return 0.0
            
        # Weighted scoring: prioritize matches in specific fields
        score = 0.0
        total_weight = 0.0
        
        # Check matches in specific metadata fields
        field_weights = {
            "environment": 1.0,
            "style_vibe": 1.0,
            "clothing_items": 0.8,
            "color_palette": 0.9,
            "detected_objects": 0.6
        }
        
        for keyword in matches:
            # Check which fields contain this keyword
            for field, weight in field_weights.items():
                field_value = metadata.get(field)
                if field_value:
                    if isinstance(field_value, list):
                        if any(keyword in self.extract_keywords(str(item)) for item in field_value):
                            score += weight
                            total_weight += weight
                            break
                    elif isinstance(field_value, str):
                        if keyword in self.extract_keywords(field_value):
                            score += weight
                            total_weight += weight
                            break
        
        # Normalize by total possible weight
        if total_weight > 0:
            score = score / total_weight
            
        # Normalize by total matches vs possible matches
        max_possible = min(len(query_keywords), len(metadata_keywords))
        if max_possible > 0:
            match_ratio = len(matches) / max_possible
            score = (score + match_ratio) / 2
            
        return normalize_to_range(score, 0.0, 1.0)


class HybridReranker:
    """
    Evaluate multi-modal overlap signals to adjust candidate order.
    """
    
    def __init__(
        self,
        image_weight: float = 0.6,
        caption_weight: float = 0.3,
        attribute_weight: float = 0.1,
        min_similarity_threshold: float = 0.0
    ):
        """
        Initialize hybrid reranker.
        
        Args:
            image_weight: Weight for image similarity (default 0.6)
            caption_weight: Weight for caption similarity (default 0.3)
            attribute_weight: Weight for attribute match (default 0.1)
            min_similarity_threshold: Minimum similarity for considering a result
        """
        self.image_weight = image_weight
        self.caption_weight = caption_weight
        self.attribute_weight = attribute_weight
        
        # Validate weights sum to 1.0
        total = image_weight + caption_weight + attribute_weight
        if abs(total - 1.0) > 0.001:
            logger.warning(f"Weights sum to {total}, normalizing to 1.0")
            self.image_weight = image_weight / total
            self.caption_weight = caption_weight / total
            self.attribute_weight = attribute_weight / total
            
        self.min_similarity_threshold = min_similarity_threshold
        self.attribute_matcher = AttributeMatcher()
        
        logger.info(
            f"Initialized HybridReranker with weights: "
            f"Image={self.image_weight:.2f}, "
            f"Caption={self.caption_weight:.2f}, "
            f"Attribute={self.attribute_weight:.2f}"
        )
        
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores using min-max scaling.
        
        Args:
            scores: List of raw scores
            
        Returns:
            Normalized scores in range [0, 1]
        """
        if not scores:
            return []
            
        scores = np.array(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score - min_score < 1e-8:
            return [0.5] * len(scores)
            
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized.tolist()
        
    def _convert_distance_to_similarity(self, distances: List[float]) -> List[float]:
        """
        Convert L2 distances to similarity scores.
        
        Args:
            distances: L2 distance values
            
        Returns:
            Similarity scores in range [0, 1]
        """
        if not distances:
            return []
            
        # Use exponential decay: similarity = exp(-distance)
        distances = np.array(distances)
        similarities = np.exp(-distances)
        return similarities.tolist()
        
    def _compute_caption_similarity(
        self,
        query_embedding: np.ndarray,
        caption_embeddings: List[np.ndarray]
    ) -> List[float]:
        """
        Compute similarity between query and caption embeddings.
        
        Args:
            query_embedding: Query embedding
            caption_embeddings: List of caption embeddings
            
        Returns:
            List of similarity scores
        """
        scores = []
        for embedding in caption_embeddings:
            if embedding is not None and len(embedding) > 0:
                try:
                    sim = cosine_similarity(query_embedding, embedding)
                    scores.append(sim)
                except Exception:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores
        
    def _compute_image_similarity(
        self,
        query_embedding: np.ndarray,
        image_embeddings: List[np.ndarray]
    ) -> List[float]:
        """
        Compute similarity between query and image embeddings.
        
        Args:
            query_embedding: Query embedding
            image_embeddings: List of image embeddings
            
        Returns:
            List of similarity scores
        """
        scores = []
        for embedding in image_embeddings:
            if embedding is not None and len(embedding) > 0:
                try:
                    sim = cosine_similarity(query_embedding, embedding)
                    scores.append(sim)
                except Exception:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores
        
    def _compute_attribute_scores(
        self,
        query: str,
        metadata_list: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Compute attribute match scores for each candidate.
        
        Args:
            query: User query string
            metadata_list: List of metadata dictionaries
            
        Returns:
            List of attribute match scores
        """
        scores = []
        for metadata in metadata_list:
            score = self.attribute_matcher.calculate_attribute_match(query, metadata)
            scores.append(score)
        return scores
        
    def rerank(
        self,
        query_string: str,
        query_embedding: np.ndarray,
        candidate_ids: List[int],
        metadata_records: List[Dict[str, Any]],
        faiss_distances: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply hybrid scoring to rerank candidates.
        
        Args:
            query_string: Original user query
            query_embedding: Query embedding
            candidate_ids: List of candidate image IDs
            metadata_records: List of metadata records for candidates
            faiss_distances: Optional FAISS distance values
            
        Returns:
            List of ranked result dictionaries containing:
            - image_id: Image identifier
            - file_path: Path to image
            - final_score: Composite score
            - image_similarity: Image similarity score
            - caption_similarity: Caption similarity score
            - attribute_match: Attribute match score
            - caption: Generated caption
            - attributes: Metadata attributes
        """
        if not candidate_ids or not metadata_records:
            logger.warning("No candidates to rerank")
            return []
            
        if len(candidate_ids) != len(metadata_records):
            logger.warning(
                f"Candidate count mismatch: {len(candidate_ids)} vs {len(metadata_records)}"
            )
            return []
            
        # Extract embeddings from metadata
        image_embeddings = []
        caption_embeddings = []
        for record in metadata_records:
            image_embeddings.append(record.get("image_embedding"))
            caption_embeddings.append(record.get("caption_embedding"))
            
        # Compute individual scores
        # 1. Image similarity
        image_scores = self._compute_image_similarity(query_embedding, image_embeddings)
        
        # 2. Caption similarity
        caption_scores = self._compute_caption_similarity(query_embedding, caption_embeddings)
        
        # 3. Attribute match
        attributes_list = [record.get("attributes", {}) for record in metadata_records]
        attribute_scores = self._compute_attribute_scores(query_string, attributes_list)
        
        # Normalize scores to [0, 1] range
        image_scores = self._normalize_scores(image_scores)
        caption_scores = self._normalize_scores(caption_scores)
        attribute_scores = self._normalize_scores(attribute_scores)
        
        # If FAISS distances provided, use them as initial image similarity
        if faiss_distances is not None and len(faiss_distances) == len(image_scores):
            # Convert distances to similarities
            faiss_similarities = self._convert_distance_to_similarity(faiss_distances)
            # Blend with computed image similarity
            image_scores = [
                0.5 * img + 0.5 * faiss 
                for img, faiss in zip(image_scores, faiss_similarities)
            ]
            
        # Compute final scores
        final_results = []
        for i in range(len(candidate_ids)):
            # Apply hybrid weights
            final_score = (
                self.image_weight * image_scores[i] +
                self.caption_weight * caption_scores[i] +
                self.attribute_weight * attribute_scores[i]
            )
            
            # Apply threshold
            if final_score < self.min_similarity_threshold:
                continue
                
            result = {
                "image_id": candidate_ids[i],
                "file_path": metadata_records[i].get("file_path", ""),
                "final_score": final_score,
                "image_similarity": image_scores[i],
                "caption_similarity": caption_scores[i],
                "attribute_match": attribute_scores[i],
                "caption": metadata_records[i].get("caption", ""),
                "attributes": metadata_records[i].get("attributes", {})
            }
            final_results.append(result)
            
        # Sort by final score descending
        final_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        logger.debug(f"Reranked {len(final_results)} results")
        return final_results
        
    def rerank_batch(
        self,
        query_string: str,
        query_embeddings: List[np.ndarray],
        candidate_ids_list: List[List[int]],
        metadata_records_list: List[List[Dict[str, Any]]],
        faiss_distances_list: Optional[List[List[float]]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Rerank multiple queries in batch.
        
        Args:
            query_strings: List of query strings
            query_embeddings: List of query embeddings
            candidate_ids_list: List of candidate ID lists
            metadata_records_list: List of metadata record lists
            faiss_distances_list: Optional list of FAISS distance lists
            
        Returns:
            List of ranked result lists
        """
        results = []
        
        for i in range(len(query_embeddings)):
            distances = None
            if faiss_distances_list and i < len(faiss_distances_list):
                distances = faiss_distances_list[i]
                
            result = self.rerank(
                query_string=query_string,
                query_embedding=query_embeddings[i],
                candidate_ids=candidate_ids_list[i],
                metadata_records=metadata_records_list[i],
                faiss_distances=distances
            )
            results.append(result)
            
        return results