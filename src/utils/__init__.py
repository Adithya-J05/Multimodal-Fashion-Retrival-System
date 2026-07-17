"""Utilities package."""

from .helpers import (
    normalize_string,
    cosine_similarity,
    l2_distance,
    compute_sha256,
    load_json_file,
    save_json_file,
    validate_image_file,
    get_image_dimensions,
    ensure_directory_exists,
    get_file_extension,
    is_supported_image,
    batch_iterator,
    safe_divide,
    normalize_to_range,
    # Fashionpedia helpers
    build_fashionpedia_lookups,
    build_image_to_annotations,
    build_image_to_filename,
    resolve_fashionpedia_metadata
)

from .logger import setup_logger, get_logger, LoggerContext
from .evaluator import Evaluator, EvaluationMetrics

__all__ = [
    # Helpers
    'normalize_string',
    'cosine_similarity',
    'l2_distance',
    'compute_sha256',
    'load_json_file',
    'save_json_file',
    'validate_image_file',
    'get_image_dimensions',
    'ensure_directory_exists',
    'get_file_extension',
    'is_supported_image',
    'batch_iterator',
    'safe_divide',
    'normalize_to_range',
    # Fashionpedia helpers
    'build_fashionpedia_lookups',
    'build_image_to_annotations',
    'build_image_to_filename',
    'resolve_fashionpedia_metadata',
    # Logger
    'setup_logger',
    'get_logger',
    'LoggerContext',
    # Evaluator
    'Evaluator',
    'EvaluationMetrics'
]