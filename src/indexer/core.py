"""Main orchestration driver for index workflows."""

import os
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from ..utils import (
    get_logger, 
    ensure_directory_exists, 
    validate_image_file,
    is_supported_image,
    load_json_file,
    save_json_file,
    resolve_fashionpedia_metadata
)
from .extractors import ExtractorManager, DeviceManager
from .storage import MetadataStore, VectorIndex

logger = get_logger(__name__)


class IndexingPipeline:
    """
    Main indexing pipeline orchestrating image validation, feature extraction,
    and persistent storage.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        extractor_manager: Optional[ExtractorManager] = None,
        metadata_store: Optional[MetadataStore] = None,
        vector_index: Optional[VectorIndex] = None
    ):
        """
        Initialize indexing pipeline.
        
        Args:
            config: Configuration dictionary
            extractor_manager: Extractor manager instance
            metadata_store: Metadata store instance
            vector_index: Vector index instance
        """
        self.config = config
        self.device_manager = DeviceManager(
            preferred_device=config.get("device", {}).get("preferred", "auto"),
            fallback_to_cpu=config.get("device", {}).get("fallback_to_cpu", True)
        )
        
        if extractor_manager is None:
            extractor_manager = ExtractorManager(
                fashion_clip_config=config.get("models", {}).get("fashion_clip", {}),
                florence_config=config.get("models", {}).get("florence_2", {}),
                device_manager=self.device_manager
            )
        self.extractor_manager = extractor_manager
        
        paths = config.get("paths", {})
        if metadata_store is None:
            metadata_store = MetadataStore(
                paths.get("metadata_db", "data/processed/metadata.db")
            )
        self.metadata_store = metadata_store
        
        if vector_index is None:
            vector_index = VectorIndex(
                index_path=paths.get("index_file", "data/processed/vectors.index"),
                dimension=config.get("index", {}).get("dimension", 512),
                index_type=config.get("index", {}).get("type", "FlatL2"),
                metric=config.get("index", {}).get("metric", "L2")
            )
        self.vector_index = vector_index
        
        self.annotation_path = config.get("fashionpedia", {}).get("annotation_path")
        self.use_fashionpedia = self.annotation_path is not None and os.path.exists(self.annotation_path)
        
        if self.use_fashionpedia:
            logger.info(f"Using Fashionpedia annotations from {self.annotation_path}")
        else:
            logger.warning("No Fashionpedia annotation file found, using Florence-2 only")
        
        self.manifest_path = paths.get("manifest_file", "data/processed/manifest.json")
        self.raw_data_dir = paths.get("raw_data", "data/raw")
        
    def run(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Execute full indexing pipeline.
        
        Args:
            force_reindex: Force reindexing even if manifest exists
            
        Returns:
            Dictionary with indexing statistics
        """
        logger.info("Starting indexing pipeline")
        
        # Validate and collect images
        image_paths = self._collect_images()
        if not image_paths:
            logger.error("No valid images found")
            return {"status": "failed", "reason": "no_images"}
        
        logger.info(f"Found {len(image_paths)} valid images")
        
        # Load existing manifest if available
        existing_manifest = self._load_manifest()
        
        # Determine which images need processing
        images_to_process = []
        if existing_manifest and not force_reindex:
            processed_paths = set(existing_manifest.get("processed_images", []))
            images_to_process = [p for p in image_paths if p not in processed_paths]
        else:
            images_to_process = image_paths
        
        logger.info(f"Processing {len(images_to_process)} new images")
        
        if not images_to_process:
            logger.info("All images already processed")
            return self._collect_stats()
        
        # Resolve Fashionpedia metadata for all images
        fashionpedia_data = {}
        if self.use_fashionpedia:
            fashionpedia_data = resolve_fashionpedia_metadata(
                self.annotation_path,
                images_to_process
            )
            logger.info(f"Resolved Fashionpedia metadata for {len(fashionpedia_data)} images")
        
        # Process images in batches
        batch_size = self.config.get("models", {}).get("fashion_clip", {}).get("batch_size", 32)
        processed_count = 0
        failed_count = 0
        
        for batch_start in tqdm(
            range(0, len(images_to_process), batch_size),
            desc="Processing images"
        ):
            batch = images_to_process[batch_start:batch_start + batch_size]
            
            try:
                # Extract features
                extracted_data = self.extractor_manager.extract_batch(batch, fashionpedia_data)
                
                # Store each item
                for data in extracted_data:
                    try:
                        # Get Fashionpedia metadata for this image
                        fp_meta = fashionpedia_data.get(data["image_path"], {})
                        
                        # Store in metadata database
                        image_id = self.metadata_store.add_record(
                            file_path=data["image_path"],
                            caption=data["caption"],
                            attributes=data["attributes"],
                            image_embedding=data["image_embedding"],
                            caption_embedding=data["caption_embedding"],
                            fashionpedia_image_id=fp_meta.get("fashionpedia_image_id"),
                            has_fashionpedia_annotations=fp_meta.get("has_annotations", False)
                        )
                        
                        # Add to vector index
                        self.vector_index.add_embedding(
                            image_id,
                            data["image_embedding"]
                        )
                        
                        processed_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to store {data['image_path']}: {e}")
                        failed_count += 1
                        
            except Exception as e:
                logger.error(f"Failed to process batch: {e}")
                failed_count += len(batch)
        
        # Save vector index and manifest
        self.vector_index.save()
        self._update_manifest(image_paths)
        
        stats = {
            "status": "completed",
            "total_images": len(image_paths),
            "processed": processed_count,
            "failed": failed_count,
            "fashionpedia_used": self.use_fashionpedia
        }
        
        logger.info(f"Indexing completed: {stats}")
        return stats
    
    def _collect_images(self) -> List[str]:
        """Collect and validate all images in raw data directory."""
        valid_images = []
        
        if not os.path.exists(self.raw_data_dir):
            logger.warning(f"Raw data directory not found: {self.raw_data_dir}")
            return []
        
        for root, _, files in os.walk(self.raw_data_dir):
            for file in files:
                filepath = os.path.join(root, file)
                
                if not is_supported_image(filepath):
                    continue
                    
                if validate_image_file(filepath):
                    valid_images.append(filepath)
                else:
                    logger.debug(f"Invalid image: {filepath}")
        
        return valid_images
    
    def _load_manifest(self) -> Optional[Dict[str, Any]]:
        """Load existing manifest file."""
        if os.path.exists(self.manifest_path):
            try:
                return load_json_file(self.manifest_path)
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
        return None
    
    def _update_manifest(self, processed_images: List[str]) -> None:
        """Update manifest with processed images."""
        manifest = self._load_manifest() or {
            "version": "1.0",
            "processed_images": []
        }
        
        existing = set(manifest["processed_images"])
        existing.update(processed_images)
        manifest["processed_images"] = list(existing)
        manifest["last_run"] = {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "fashionpedia_used": self.use_fashionpedia
        }
        
        save_json_file(manifest, self.manifest_path)
        logger.info(f"Updated manifest with {len(existing)} processed images")
    
    def _collect_stats(self) -> Dict[str, Any]:
        """Collect current indexing statistics."""
        manifest = self._load_manifest()
        if not manifest:
            return {"status": "no_manifest", "total_images": 0}
        
        return {
            "status": "completed",
            "total_images": len(manifest.get("processed_images", [])),
            "fashionpedia_used": self.use_fashionpedia,
            "last_run": manifest.get("last_run")
        }