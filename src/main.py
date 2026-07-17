"""Core CLI entrypoint orchestrating pipeline execution."""

import os
import sys
import argparse
import yaml
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_logger, ensure_directory_exists
from src.utils.evaluator import Evaluator
from src.indexer.core import IndexingPipeline
from src.retriever.core import RetrievalPipeline

logger = get_logger(__name__)


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    logger.info(f"Loaded configuration from {config_path}")
    return config


def setup_environment(config: Dict[str, Any]) -> None:
    """
    Setup environment directories.
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get("paths", {})
    directories = [
        paths.get("raw_data", "data/raw"),
        paths.get("processed_data", "data/processed"),
        paths.get("evaluation_output", "data/processed/evaluation")
    ]
    
    for directory in directories:
        ensure_directory_exists(directory)
        logger.debug(f"Ensured directory: {directory}")


def run_indexing(config: Dict[str, Any], force: bool = False) -> None:
    """
    Run indexing pipeline.
    
    Args:
        config: Configuration dictionary
        force: Force reindexing
    """
    logger.info("Starting indexing pipeline...")
    
    setup_environment(config)
    
    pipeline = IndexingPipeline(config)
    result = pipeline.run(force_reindex=force)
    
    logger.info(f"Indexing completed: {result}")
    
    if result.get("status") != "completed":
        sys.exit(1)


def run_search(config: Dict[str, Any], query: str, k: int = None) -> None:
    """
    Run search pipeline.
    
    Args:
        config: Configuration dictionary
        query: Query string
        k: Number of results
    """
    logger.info(f"Searching for: '{query}'")
    
    setup_environment(config)
    
    pipeline = RetrievalPipeline(config)
    results = pipeline.search(query, k)
    
    print("\n" + "=" * 100)
    print(f"SEARCH RESULTS: '{query}'")
    print("=" * 100)
    
    if not results:
        print("No results found.")
        return
        
    print(f"\n{'Rank':<6} {'Image ID':<10} {'Final Score':<14} {'File Path':<50}")
    print("-" * 100)
    
    for rank, result in enumerate(results[:k or 10], 1):
        file_path = result.get("file_path", "unknown")
        file_name = os.path.basename(file_path)[:45] + "..." if len(file_path) > 48 else file_path
        
        print(
            f"{rank:<6} "
            f"{result.get('image_id', '?'):<10} "
            f"{result.get('final_score', 0.0):<14.6f} "
            f"{file_name:<50}"
        )
    
    print("\n" + "=" * 100)


def run_evaluation(config: Dict[str, Any]) -> None:
    """
    Run evaluation suite.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting evaluation suite...")
    
    setup_environment(config)
    
    retrieval_pipeline = RetrievalPipeline(config)
    evaluator = Evaluator(config, retrieval_pipeline)
    
    summary = evaluator.run_evaluation()
    
    if summary.get("failed", 0) > 0:
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fashion Context Retrieval System"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Run indexing pipeline")
    index_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindexing even if manifest exists"
    )
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Run search")
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query string"
    )
    search_parser.add_argument(
        "--k",
        type=int,
        help="Number of results to return"
    )
    
    # Evaluate command
    subparsers.add_parser("evaluate", help="Run evaluation suite")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        config = load_config(args.config)
        
        if args.command == "index":
            run_indexing(config, args.force)
        elif args.command == "search":
            run_search(config, args.query, args.k)
        elif args.command == "evaluate":
            run_evaluation(config)
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()