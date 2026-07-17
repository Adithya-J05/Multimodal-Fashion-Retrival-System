"""FashionCLIP and Florence-2 model wrappers for feature extraction."""
import os
os.environ["FLASH_ATTENTION_DISABLE"] = "1"
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from PIL import Image
from transformers import (
    CLIPModel, CLIPProcessor,
    AutoModelForCausalLM, AutoProcessor
)

from ..utils import get_logger

logger = get_logger(__name__)


class DeviceManager:
    """Manage device configuration and routing."""
    
    def __init__(self, preferred_device: str = "auto", fallback_to_cpu: bool = True):
        self.preferred_device = preferred_device
        self.fallback_to_cpu = fallback_to_cpu
        self._device = None
        
    def get_device(self) -> torch.device:
        if self._device is not None:
            return self._device
            
        if self.preferred_device == "cpu":
            self._device = torch.device("cpu")
            return self._device
            
        if self.preferred_device == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
            return self._device
            
        if self.preferred_device == "mps" and torch.backends.mps.is_available():
            self._device = torch.device("mps")
            return self._device
            
        if self.preferred_device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                return self._device
            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
                return self._device
                
        if self.fallback_to_cpu:
            self._device = torch.device("cpu")
            return self._device
            
        raise RuntimeError(f"No suitable device found for {self.preferred_device}")


class FashionCLIPExtractor:
    """FashionCLIP wrapper using HuggingFace Transformers."""
    
    def __init__(
        self,
        model_name: str = "patrickjohncyh/fashion-clip",
        embedding_dim: int = 512,
        device_manager: Optional[DeviceManager] = None,
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.device_manager = device_manager or DeviceManager()
        self._model = None
        self._processor = None
        
    def _initialize_model(self):
        if self._model is not None:
            return
            
        try:
            device = self.device_manager.get_device()
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name).to(device)
            self._model.eval()
            logger.info(f"Initialized FashionCLIP on {device}")
        except Exception as e:
            logger.error(f"Failed to initialize FashionCLIP: {e}")
            raise
            
    def _get_embedding(self, outputs):
        """Extract embedding from CLIP outputs (handles both tensor and BaseModelOutput)."""
        if hasattr(outputs, 'pooler_output'):
            return outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state[:, 0, :]
        else:
            return outputs
            
    def extract_image_embedding(self, image_path: str) -> np.ndarray:
        self._initialize_model()
        device = self.device_manager.get_device()
        
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = self._model.get_image_features(**inputs)
                embedding = self._get_embedding(outputs)
                embedding = embedding.cpu().numpy().astype(np.float32)
            
            if embedding.shape[1] != self.embedding_dim:
                raise ValueError(f"Expected {self.embedding_dim}, got {embedding.shape[1]}")
                
            return embedding[0]
        except Exception as e:
            logger.error(f"Failed to extract embedding from {image_path}: {e}")
            raise
            
    def extract_batch_image_embeddings(self, image_paths: List[str]) -> np.ndarray:
        self._initialize_model()
        device = self.device_manager.get_device()
        
        if not image_paths:
            return np.array([])
            
        try:
            images = [Image.open(path).convert("RGB") for path in image_paths]
            inputs = self._processor(images=images, return_tensors="pt")
            
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = self._model.get_image_features(**inputs)
                embeddings = self._get_embedding(outputs)
                embeddings = embeddings.cpu().numpy().astype(np.float32)
            
            if embeddings.shape[1] != self.embedding_dim:
                raise ValueError(f"Expected {self.embedding_dim}, got {embeddings.shape[1]}")
                
            return embeddings
        except Exception as e:
            logger.error(f"Failed to extract batch embeddings: {e}")
            raise
            
    def encode_text(self, text: str) -> np.ndarray:
        self._initialize_model()
        device = self.device_manager.get_device()
        
        try:
            inputs = self._processor(text=text, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = self._model.get_text_features(**inputs)
                embedding = self._get_embedding(outputs)
                embedding = embedding.cpu().numpy().astype(np.float32)
            
            if embedding.shape[1] != self.embedding_dim:
                raise ValueError(f"Expected {self.embedding_dim}, got {embedding.shape[1]}")
                
            return embedding[0]
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise
            
    def encode_batch_texts(self, texts: List[str]) -> np.ndarray:
        self._initialize_model()
        device = self.device_manager.get_device()
        
        if not texts:
            return np.array([])
            
        try:
            inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = self._model.get_text_features(**inputs)
                embeddings = self._get_embedding(outputs)
                embeddings = embeddings.cpu().numpy().astype(np.float32)
            
            if embeddings.shape[1] != self.embedding_dim:
                raise ValueError(f"Expected {self.embedding_dim}, got {embeddings.shape[1]}")
                
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode batch texts: {e}")
            raise


class Florence2Extractor:
    """
    Florence-2 wrapper for detailed image captioning with environment context.
    Optimized for GPU with half precision.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        device_manager: Optional[DeviceManager] = None,
        max_length: int = 256,
        batch_size: int = 4,
        task_prompt: str = "<DETAILED_CAPTION>"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.task_prompt = task_prompt
        self.device_manager = device_manager or DeviceManager()
        self._model = None
        self._processor = None
        
    def _initialize_model(self):
        if self._model is not None:
            return
            
        try:
            device = self.device_manager.get_device()
            
            # Load processor with trust_remote_code
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Use half precision on GPU to save memory
            if device.type == 'cuda':
                torch_dtype = torch.float16
                logger.info("Using float16 for Florence-2 on GPU")
            else:
                torch_dtype = torch.float32
                logger.info("Using float32 for Florence-2 on CPU")
            
            # Load model with appropriate dtype
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                attn_implementation="eager"   # <-- ADD THIS LINE
            ).to(device)
            
            self._model.eval()
            logger.info(f"Initialized Florence-2 on {device} with dtype {torch_dtype}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Florence-2: {e}")
            raise
            
    def extract_detailed_caption(self, image_path: str) -> str:
        """Extract detailed caption from a single image."""
        self._initialize_model()
        device = self.device_manager.get_device()
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Prepare inputs with task prompt
            inputs = self._processor(
                text=self.task_prompt,
                images=image,
                return_tensors="pt"
            )
            
            with torch.inference_mode():
                inputs = {k: v.to(device) for k, v in inputs.items()}
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=self.max_length,
                    num_beams=3,
                    early_stopping=True,
                    do_sample=False
                )
                
            caption = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            return caption
            
        except Exception as e:
            logger.error(f"Failed to extract caption from {image_path}: {e}")
            return ""
            
    def extract_batch_captions(self, image_paths: List[str]) -> List[str]:
        """
        Extract detailed captions for a batch of images.
        Processes in smaller sub-batches to manage GPU memory.
        """
        self._initialize_model()
        results = []
        
        # Process in micro-batches of size self.batch_size
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i+self.batch_size]
            batch_captions = []
            
            # Process each image individually to avoid OOM
            for path in batch_paths:
                try:
                    caption = self.extract_detailed_caption(path)
                    batch_captions.append(caption)
                except Exception as e:
                    logger.error(f"Failed for {path}: {e}")
                    batch_captions.append("")
            
            results.extend(batch_captions)
            
            # Clear GPU cache after each micro-batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return results
        
    def extract_environment_from_caption(self, caption: str) -> str:
        """Extract environment context from caption."""
        if not caption:
            return "unknown"
            
        text = caption.lower()
        environment_keywords = {
            "office": ["office", "desk", "computer", "laptop", "meeting", "conference", "workplace"],
            "street": ["street", "road", "sidewalk", "pavement", "building", "city", "urban"],
            "park": ["park", "garden", "tree", "grass", "flower", "bench", "nature"],
            "home": ["home", "room", "living room", "bedroom", "kitchen", "house", "apartment"],
        }
        
        for env, keywords in environment_keywords.items():
            if any(kw in text for kw in keywords):
                return env
        return "unknown"


class ExtractorManager:
    """
    Manager orchestrating FashionCLIP and Florence-2 extractors.
    """
    
    def __init__(
        self,
        fashion_clip_config: Optional[Dict[str, Any]] = None,
        florence_config: Optional[Dict[str, Any]] = None,
        device_manager: Optional[DeviceManager] = None
    ):
        self.device_manager = device_manager or DeviceManager()
        
        fashion_clip_config = fashion_clip_config or {}
        florence_config = florence_config or {}
        
        self.fashion_clip = FashionCLIPExtractor(
            model_name=fashion_clip_config.get("model_name", "patrickjohncyh/fashion-clip"),
            embedding_dim=fashion_clip_config.get("embedding_dim", 512),
            device_manager=self.device_manager,
            batch_size=fashion_clip_config.get("batch_size", 32)
        )
        
        self.florence = Florence2Extractor(
            model_name=florence_config.get("model_name", "microsoft/Florence-2-base"),
            device_manager=self.device_manager,
            max_length=florence_config.get("max_length", 256),
            batch_size=florence_config.get("batch_size", 4),
            task_prompt=florence_config.get("task_prompt", "<DETAILED_CAPTION>")
        )
        
        logger.info("Initialized ExtractorManager with Florence-2 for captioning")
        
    def extract_all(self, image_path: str, fashionpedia_attributes: Dict[str, Any] = None) -> Dict[str, Any]:
        fashionpedia_attributes = fashionpedia_attributes or {}
        
        try:
            image_embedding = self.fashion_clip.extract_image_embedding(image_path)
            caption = self.florence.extract_detailed_caption(image_path)
            caption_embedding = self.fashion_clip.encode_text(caption)
            environment = self.florence.extract_environment_from_caption(caption)
            
            attributes = {
                "environment": environment,
                "clothing_items": fashionpedia_attributes.get("clothing_items", []),
                "color_palette": fashionpedia_attributes.get("color_palette", []),
                "style_vibe": fashionpedia_attributes.get("style_vibe", "unknown"),
                "detected_objects": fashionpedia_attributes.get("detected_objects", []),
                "fashionpedia_image_id": fashionpedia_attributes.get("fashionpedia_image_id"),
                "has_fashionpedia_annotations": fashionpedia_attributes.get("has_fashionpedia_annotations", False)
            }
            
            return {
                "image_embedding": image_embedding,
                "caption": caption,
                "caption_embedding": caption_embedding,
                "attributes": attributes
            }
        except Exception as e:
            logger.error(f"Failed to extract features from {image_path}: {e}")
            raise
            
    def extract_batch(self, image_paths: List[str], fashionpedia_data: Dict[str, Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not image_paths:
            return []
            
        fashionpedia_data = fashionpedia_data or {}
        results = []
        
        # Extract image embeddings in batch
        embeddings = self.fashion_clip.extract_batch_image_embeddings(image_paths)
        
        # Extract captions in batch (with memory management)
        captions = self.florence.extract_batch_captions(image_paths)
        
        # Encode captions in batch
        caption_embeddings = self.fashion_clip.encode_batch_texts(captions)
        
        for i, (image_path, embedding, caption, caption_embedding) in enumerate(
            zip(image_paths, embeddings, captions, caption_embeddings)
        ):
            fp_meta = fashionpedia_data.get(image_path, {})
            environment = self.florence.extract_environment_from_caption(caption)
            
            attributes = {
                "environment": environment,
                "clothing_items": fp_meta.get("clothing_items", []),
                "color_palette": fp_meta.get("color_palette", []),
                "style_vibe": fp_meta.get("style_vibe", "unknown"),
                "detected_objects": fp_meta.get("detected_objects", []),
                "fashionpedia_image_id": fp_meta.get("fashionpedia_image_id"),
                "has_fashionpedia_annotations": fp_meta.get("has_fashionpedia_annotations", False)
            }
            
            results.append({
                "image_path": image_path,
                "image_embedding": embedding,
                "caption": caption,
                "caption_embedding": caption_embedding,
                "attributes": attributes
            })
            
        return results