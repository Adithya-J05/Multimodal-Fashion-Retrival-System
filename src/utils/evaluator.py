"""Evaluation framework for retrieval system validation."""

import os
import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from .logger import get_logger
from .helpers import normalize_string, ensure_directory_exists, save_json_file

logger = get_logger(__name__)


class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    def __init__(self):
        self.query = ""
        self.rank = 0
        self.image_id = None
        self.file_path = ""
        self.final_score = 0.0
        self.image_similarity = 0.0
        self.caption_similarity = 0.0
        self.attribute_match = 0.0
        self.attributes_found = []
        self.expected_attributes = []
        self.precision_at_k = 0.0
        self.recall_at_k = 0.0


class Evaluator:
    """
    Evaluation framework for validating retrieval system performance.
    """
    
    # Mandatory evaluation queries
    EVALUATION_QUERIES = [
        {
            "id": 1,
            "query": "A person in a bright yellow raincoat.",
            "expected_attributes": ["yellow", "raincoat"],
            "expected_environment": "street"
        },
        {
            "id": 2,
            "query": "Professional business attire inside a modern office.",
            "expected_attributes": ["business", "office"],
            "expected_environment": "office"
        },
        {
            "id": 3,
            "query": "Someone wearing a blue shirt sitting on a park bench.",
            "expected_attributes": ["blue", "shirt", "park"],
            "expected_environment": "park"
        },
        {
            "id": 4,
            "query": "Casual weekend outfit for a city walk.",
            "expected_attributes": ["casual", "city"],
            "expected_environment": "street"
        },
        {
            "id": 5,
            "query": "A red tie and a white shirt in a formal setting.",
            "expected_attributes": ["red", "tie", "white", "shirt", "formal"],
            "expected_environment": "office"
        }
    ]
    
    def __init__(
        self,
        config: Dict[str, Any],
        retrieval_pipeline,
        output_dir: str = "data/processed/evaluation"
    ):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
            retrieval_pipeline: Initialized RetrievalPipeline instance
            output_dir: Directory for evaluation outputs
        """
        self.config = config
        self.retrieval_pipeline = retrieval_pipeline
        self.output_dir = output_dir
        ensure_directory_exists(output_dir)
        
        self.top_k = config.get("retrieval", {}).get("default_k", 5)
        self.results = []
        
        logger.info(f"Initialized Evaluator with top_k={self.top_k}")
        
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run complete evaluation suite.
        
        Returns:
            Evaluation results dictionary
        """
        logger.info("Starting evaluation suite")
        
        all_results = []
        query_results = []
        
        for eval_query in self.EVALUATION_QUERIES:
            query_text = eval_query["query"]
            expected_attrs = eval_query.get("expected_attributes", [])
            expected_env = eval_query.get("expected_environment", "")
            
            logger.info(f"Processing query {eval_query['id']}: '{query_text}'")
            
            try:
                # Execute search
                search_results = self.retrieval_pipeline.search(
                    query=query_text,
                    k=self.top_k
                )
                
                # Evaluate results
                evaluated = self._evaluate_query_results(
                    query_text=query_text,
                    query_id=eval_query["id"],
                    results=search_results,
                    expected_attributes=expected_attrs,
                    expected_environment=expected_env
                )
                
                query_results.append(evaluated)
                all_results.extend(evaluated.get("detailed_results", []))
                
            except Exception as e:
                logger.error(f"Failed to evaluate query {eval_query['id']}: {e}")
                query_results.append({
                    "query_id": eval_query["id"],
                    "query": query_text,
                    "status": "failed",
                    "error": str(e),
                    "detailed_results": []
                })
        
        # Compile summary
        summary = self._compile_summary(query_results)
        
        # Save results
        self._save_results(summary, query_results, all_results)
        
        # Print dashboard
        self._print_dashboard(query_results)
        
        return summary
        
    def _evaluate_query_results(
        self,
        query_text: str,
        query_id: int,
        results: List[Dict[str, Any]],
        expected_attributes: List[str],
        expected_environment: str
    ) -> Dict[str, Any]:
        """
        Evaluate results for a single query.
        
        Args:
            query_text: Query string
            query_id: Query identifier
            results: Retrieved results
            expected_attributes: Expected attribute list
            expected_environment: Expected environment
            
        Returns:
            Evaluation results dictionary
        """
        evaluated_results = []
        
        # Normalize expected attributes
        expected_attrs_norm = [normalize_string(attr) for attr in expected_attributes]
        
        for rank, result in enumerate(results, 1):
            attributes = result.get("attributes", {})
            clothing_items = attributes.get("clothing_items", [])
            color_palette = attributes.get("color_palette", [])
            environment = attributes.get("environment", "unknown")
            
            # Check attribute matches
            found_attributes = []
            for expected_attr in expected_attrs_norm:
                # Check in clothing items
                if any(expected_attr in normalize_string(item) for item in clothing_items):
                    found_attributes.append(expected_attr)
                # Check in color palette
                elif any(expected_attr in normalize_string(color) for color in color_palette):
                    found_attributes.append(expected_attr)
                # Check in environment
                elif expected_attr in normalize_string(environment):
                    found_attributes.append(expected_attr)
            
            # Check environment match
            environment_match = False
            if expected_environment:
                environment_match = normalize_string(environment) == normalize_string(expected_environment)
            
            evaluated = {
                "rank": rank,
                "image_id": result.get("image_id"),
                "file_path": result.get("file_path", ""),
                "final_score": result.get("final_score", 0.0),
                "image_similarity": result.get("image_similarity", 0.0),
                "caption_similarity": result.get("caption_similarity", 0.0),
                "attribute_match": result.get("attribute_match", 0.0),
                "attributes_found": found_attributes,
                "environment_found": environment,
                "environment_match": environment_match,
                "clothing_items": clothing_items,
                "color_palette": color_palette
            }
            evaluated_results.append(evaluated)
        
        # Calculate metrics
        total_expected = len(expected_attrs_norm)
        if total_expected > 0:
            # Precision: ratio of found attributes to expected
            found_count = sum(1 for r in evaluated_results for attr in r.get("attributes_found", []))
            precision = found_count / (total_expected * len(evaluated_results)) if len(evaluated_results) > 0 else 0.0
            
            # Recall: unique found attributes / total expected
            unique_found = set()
            for r in evaluated_results:
                unique_found.update(r.get("attributes_found", []))
            recall = len(unique_found) / total_expected if total_expected > 0 else 0.0
        else:
            precision = 0.0
            recall = 0.0
        
        return {
            "query_id": query_id,
            "query": query_text,
            "status": "success",
            "total_results": len(evaluated_results),
            "precision": precision,
            "recall": recall,
            "detailed_results": evaluated_results,
            "expected_attributes": expected_attrs_norm,
            "expected_environment": expected_environment
        }
        
    def _compile_summary(self, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compile summary statistics.
        
        Args:
            query_results: List of query evaluation results
            
        Returns:
            Summary dictionary
        """
        successful = [q for q in query_results if q.get("status") == "success"]
        failed = [q for q in query_results if q.get("status") != "success"]
        
        avg_precision = 0.0
        avg_recall = 0.0
        
        if successful:
            avg_precision = sum(q.get("precision", 0.0) for q in successful) / len(successful)
            avg_recall = sum(q.get("recall", 0.0) for q in successful) / len(successful)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(query_results),
            "successful": len(successful),
            "failed": len(failed),
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "query_results": query_results
        }
        
    def _save_results(
        self,
        summary: Dict[str, Any],
        query_results: List[Dict[str, Any]],
        all_results: List[Dict[str, Any]]
    ) -> None:
        """
        Save evaluation results to files.
        
        Args:
            summary: Summary dictionary
            query_results: Query results list
            all_results: All detailed results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_path = os.path.join(self.output_dir, f"summary_{timestamp}.json")
        save_json_file(summary, summary_path)
        
        # Save query results
        query_path = os.path.join(self.output_dir, f"query_results_{timestamp}.json")
        save_json_file(query_results, query_path)
        
        # Save detailed results
        detailed_path = os.path.join(self.output_dir, f"detailed_results_{timestamp}.json")
        save_json_file(all_results, detailed_path)
        
        logger.info(f"Saved evaluation results to {self.output_dir}")
        
    def _print_dashboard(self, query_results: List[Dict[str, Any]]) -> None:
        """
        Print formatted dashboard to console.
        
        Args:
            query_results: List of query evaluation results
        """
        print("\n" + "=" * 120)
        print("RETRIEVAL EVALUATION DASHBOARD")
        print("=" * 120)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 120)
        
        # Header row
        header = (
            f"{'Query ID':<10} "
            f"{'Query Text':<40} "
            f"{'Status':<10} "
            f"{'Results':<8} "
            f"{'Precision':<12} "
            f"{'Recall':<12}"
        )
        print(header)
        print("-" * 120)
        
        # Query rows
        for qr in query_results:
            query_id = qr.get("query_id", "?")
            query_text = qr.get("query", "")[:37] + "..." if len(qr.get("query", "")) > 40 else qr.get("query", "")
            status = qr.get("status", "unknown")
            total_results = qr.get("total_results", 0)
            precision = qr.get("precision", 0.0)
            recall = qr.get("recall", 0.0)
            
            print(
                f"{query_id:<10} "
                f"{query_text:<40} "
                f"{status:<10} "
                f"{total_results:<8} "
                f"{precision:<12.4f} "
                f"{recall:<12.4f}"
            )
        
        print("-" * 120)
        
        # Detailed results
        print("\n" + "=" * 120)
        print("DETAILED RESULTS (Top-5 per query)")
        print("=" * 120)
        
        for qr in query_results:
            if qr.get("status") != "success":
                continue
                
            query_id = qr.get("query_id", "?")
            query_text = qr.get("query", "")[:50] + "..." if len(qr.get("query", "")) > 53 else qr.get("query", "")
            
            print(f"\nQuery {query_id}: {query_text}")
            print("-" * 120)
            
            # Sub-header for detailed results
            detail_header = (
                f"{'Rank':<6} "
                f"{'Image ID':<10} "
                f"{'Final Score':<14} "
                f"{'Image Sim':<12} "
                f"{'Caption Sim':<13} "
                f"{'Attr Match':<12} "
                f"{'Attributes Found':<30} "
                f"{'File Path':<30}"
            )
            print(detail_header)
            print("-" * 120)
            
            detailed = qr.get("detailed_results", [])
            if not detailed:
                print("  No results returned")
                continue
                
            for result in detailed[:5]:
                rank = result.get("rank", "?")
                image_id = result.get("image_id", "?")
                final_score = result.get("final_score", 0.0)
                image_sim = result.get("image_similarity", 0.0)
                caption_sim = result.get("caption_similarity", 0.0)
                attr_match = result.get("attribute_match", 0.0)
                attrs_found = ", ".join(result.get("attributes_found", [])[:5])
                file_path = result.get("file_path", "").split("/")[-1][:27] + "..." if len(result.get("file_path", "")) > 30 else result.get("file_path", "")
                
                print(
                    f"{rank:<6} "
                    f"{image_id:<10} "
                    f"{final_score:<14.6f} "
                    f"{image_sim:<12.6f} "
                    f"{caption_sim:<13.6f} "
                    f"{attr_match:<12.6f} "
                    f"{attrs_found:<30} "
                    f"{file_path:<30}"
                )
        
        print("\n" + "=" * 120)
        
        # Summary metrics
        successful = [q for q in query_results if q.get("status") == "success"]
        if successful:
            avg_precision = sum(q.get("precision", 0.0) for q in successful) / len(successful)
            avg_recall = sum(q.get("recall", 0.0) for q in successful) / len(successful)
            
            print(f"SUMMARY METRICS:")
            print(f"  Total Queries: {len(query_results)}")
            print(f"  Successful: {len(successful)}")
            print(f"  Failed: {len(query_results) - len(successful)}")
            print(f"  Average Precision: {avg_precision:.4f}")
            print(f"  Average Recall: {avg_recall:.4f}")
        
        print("\n" + "=" * 120)
        print(f"Detailed results saved to: {self.output_dir}")
        print("=" * 120 + "\n")