"""Common mathematical operations, string normalizers, IO actions."""

import json
import os
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import numpy as np
from PIL import Image


def normalize_string(text: str) -> str:
    """
    Normalize string for consistent comparison.
    
    Args:
        text: Input string
    
    Returns:
        Normalized lowercase string with stripped whitespace
    """
    if not text:
        return ""
    return text.lower().strip()


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity score in range [0, 1]
    """
    if vec1 is None or vec2 is None:
        return 0.0
        
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def l2_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute L2 distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        L2 distance value
    """
    if vec1 is None or vec2 is None:
        return float('inf')
    
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
    
    return float(np.linalg.norm(vec1 - vec2))


def compute_sha256(filepath: str) -> str:
    """
    Compute SHA256 hash of a file.
    
    Args:
        filepath: Path to the file
    
    Returns:
        SHA256 hex digest
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_json_file(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file with error handling.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Dictionary containing JSON data
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")


def save_json_file(data: Dict[str, Any], filepath: str, indent: int = 2) -> None:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation level
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def validate_image_file(filepath: str) -> bool:
    """
    Validate image file can be opened and has reasonable dimensions.
    
    Args:
        filepath: Path to image file
    
    Returns:
        True if image is valid, False otherwise
    """
    try:
        with Image.open(filepath) as img:
            width, height = img.size
            
            # Check minimum dimensions
            if width < 64 or height < 64:
                return False
            
            # Check maximum dimensions
            if width > 4096 or height > 4096:
                return False
            
            # Verify image can be loaded properly
            img.verify()
            return True
            
    except Exception:
        return False


def get_image_dimensions(filepath: str) -> Tuple[int, int]:
    """
    Get image dimensions without loading full image.
    
    Args:
        filepath: Path to image file
    
    Returns:
        Tuple of (width, height)
    """
    try:
        with Image.open(filepath) as img:
            return img.size
    except Exception:
        return (0, 0)


def ensure_directory_exists(path: str) -> None:
    """
    Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_file_extension(filepath: str) -> str:
    """
    Get file extension in lowercase.
    
    Args:
        filepath: File path
    
    Returns:
        File extension including dot (e.g., '.jpg')
    """
    return os.path.splitext(filepath)[1].lower()


def is_supported_image(filepath: str) -> bool:
    """
    Check if file is a supported image based on extension.
    
    Args:
        filepath: File path
    
    Returns:
        True if file extension is supported
    """
    supported = ['.jpg', '.jpeg', '.png']
    return get_file_extension(filepath) in supported


def batch_iterator(items: List[Any], batch_size: int):
    """
    Yield batches from a list.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
    
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division operation avoiding division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero
    
    Returns:
        Division result or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def normalize_to_range(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize value to specified range.
    
    Args:
        value: Input value
        min_val: Minimum of target range
        max_val: Maximum of target range
    
    Returns:
        Normalized value
    """
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value


# ============ FASHIONPEDIA ANNOTATION HELPERS ============

def build_fashionpedia_lookups(annotation_path: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Build fast in-memory dictionaries for Fashionpedia category and attribute mappings.
    
    Args:
        annotation_path: Path to Fashionpedia JSON annotation file
    
    Returns:
        Tuple of (category_id_to_name, attribute_id_to_name) dictionaries
    """
    data = load_json_file(annotation_path)
    
    # Build category lookup
    category_lookup = {}
    for category in data.get("categories", []):
        cat_id = category.get("id")
        cat_name = category.get("name")
        if cat_id is not None and cat_name is not None:
            category_lookup[cat_id] = normalize_string(cat_name)
    
    # Build attribute lookup
    attribute_lookup = {}
    for attribute in data.get("attributes", []):
        attr_id = attribute.get("id")
        attr_name = attribute.get("name")
        if attr_id is not None and attr_name is not None:
            attribute_lookup[attr_id] = normalize_string(attr_name)
    
    return category_lookup, attribute_lookup


def build_image_to_annotations(annotation_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Build mapping from image_id to its annotations.
    
    Args:
        annotation_path: Path to Fashionpedia JSON annotation file
    
    Returns:
        Dictionary mapping image_id to list of annotation dictionaries
    """
    data = load_json_file(annotation_path)
    
    image_to_annotations = {}
    for annotation in data.get("annotations", []):
        image_id = annotation.get("image_id")
        if image_id is None:
            continue
            
        if image_id not in image_to_annotations:
            image_to_annotations[image_id] = []
            
        image_to_annotations[image_id].append(annotation)
    
    return image_to_annotations


def build_image_to_filename(annotation_path: str) -> Dict[int, str]:
    """
    Build mapping from image_id to file_name.
    
    Args:
        annotation_path: Path to Fashionpedia JSON annotation file
    
    Returns:
        Dictionary mapping image_id to file_name string
    """
    data = load_json_file(annotation_path)
    
    image_to_filename = {}
    for image in data.get("images", []):
        image_id = image.get("id")
        file_name = image.get("file_name")
        if image_id is not None and file_name is not None:
            image_to_filename[image_id] = file_name
    
    return image_to_filename


def resolve_fashionpedia_metadata(
    annotation_path: str,
    image_paths: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Resolve Fashionpedia metadata for a list of image paths.
    
    Args:
        annotation_path: Path to Fashionpedia JSON annotation file
        image_paths: List of image file paths (basename will be used for matching)
    
    Returns:
        Dictionary mapping image_path to metadata dict containing:
        - clothing_items: list of category names
        - color_palette: list of attribute names
        - image_id: Fashionpedia image ID
    """
    # Build lookups
    category_lookup, attribute_lookup = build_fashionpedia_lookups(annotation_path)
    image_to_filename = build_image_to_filename(annotation_path)
    image_to_annotations = build_image_to_annotations(annotation_path)
    
    # Build reverse lookup: filename -> image_id
    filename_to_image_id = {v: k for k, v in image_to_filename.items()}
    
    result = {}
    
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        
        # Find matching image_id
        image_id = filename_to_image_id.get(filename)
        if image_id is None:
            # Try without extensions or with different extensions
            base_name = os.path.splitext(filename)[0]
            for fname, img_id in filename_to_image_id.items():
                if os.path.splitext(fname)[0] == base_name:
                    image_id = img_id
                    break
        
        if image_id is None:
            result[image_path] = {
                "clothing_items": [],
                "color_palette": [],
                "image_id": None,
                "has_annotations": False
            }
            continue
        
        # Get annotations for this image
        annotations = image_to_annotations.get(image_id, [])
        
        clothing_items = set()
        color_palette = set()
        
        for annotation in annotations:
            # Resolve category
            category_id = annotation.get("category_id")
            if category_id is not None and category_id in category_lookup:
                clothing_items.add(category_lookup[category_id])
            
            # Resolve attributes
            attribute_ids = annotation.get("attribute_ids", [])
            if isinstance(attribute_ids, list):
                for attr_id in attribute_ids:
                    if attr_id in attribute_lookup:
                        # Simple heuristic: color attributes are typically first
                        attr_name = attribute_lookup[attr_id]
                        if any(color in attr_name for color in 
                               ["black", "white", "red", "blue", "green", "yellow",
                                "purple", "orange", "pink", "brown", "gray", "grey",
                                "navy", "beige", "cream", "gold", "silver", "maroon",
                                "teal", "turquoise", "magenta", "lime", "olive"]):
                            color_palette.add(attr_name)
                        else:
                            # Non-color attributes can be added to clothing items
                            # if they describe the garment
                            if any(fabric in attr_name for fabric in 
                                   ["cotton", "wool", "silk", "leather", "denim",
                                    "lace", "velvet", "cashmere", "linen"]):
                                # Could add to a materials field, but keep in clothing_items
                                # for backward compatibility
                                clothing_items.add(attr_name)
        
        result[image_path] = {
            "clothing_items": list(clothing_items),
            "color_palette": list(color_palette),
            "image_id": image_id,
            "has_annotations": len(annotations) > 0
        }
    
    return result