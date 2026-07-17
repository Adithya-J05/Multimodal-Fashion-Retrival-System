"""Unit tests for evaluation framework."""

import os
import sys
import unittest
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.evaluator import Evaluator, EvaluationMetrics


class TestEvaluator(unittest.TestCase):
    """Test evaluator functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "retrieval": {
                "default_k": 5,
                "max_k": 500
            }
        }
        
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        mock_pipeline = MockRetrievalPipeline()
        evaluator = Evaluator(self.config, mock_pipeline)
        
        self.assertEqual(evaluator.top_k, 5)
        self.assertEqual(evaluator.output_dir, "data/processed/evaluation")
        
    def test_evaluation_queries_defined(self):
        """Test evaluation queries are defined."""
        self.assertTrue(len(Evaluator.EVALUATION_QUERIES) > 0)
        self.assertEqual(len(Evaluator.EVALUATION_QUERIES), 5)
        
        for query in Evaluator.EVALUATION_QUERIES:
            self.assertIn("id", query)
            self.assertIn("query", query)
            self.assertIn("expected_attributes", query)
            self.assertIn("expected_environment", query)
            
    def test_metric_calculation(self):
        """Test metric calculation logic."""
        evaluator = Evaluator(self.config, MockRetrievalPipeline())
        
        # Test with some sample results
        results = [
            {
                "image_id": 1,
                "file_path": "test1.jpg",
                "final_score": 0.8,
                "image_similarity": 0.7,
                "caption_similarity": 0.6,
                "attribute_match": 0.5,
                "attributes": {
                    "clothing_items": ["shirt", "tie"],
                    "color_palette": ["red", "white"],
                    "environment": "office"
                }
            }
        ]
        
        expected_attrs = ["red", "white", "shirt"]
        expected_env = "office"
        
        evaluated = evaluator._evaluate_query_results(
            query_text="test query",
            query_id=1,
            results=results,
            expected_attributes=expected_attrs,
            expected_environment=expected_env
        )
        
        self.assertEqual(evaluated["status"], "success")
        self.assertEqual(evaluated["total_results"], 1)
        self.assertGreater(evaluated["precision"], 0)
        
    def test_empty_results_handling(self):
        """Test handling of empty results."""
        evaluator = Evaluator(self.config, MockRetrievalPipeline())
        
        evaluated = evaluator._evaluate_query_results(
            query_text="test query",
            query_id=1,
            results=[],
            expected_attributes=["red"],
            expected_environment="office"
        )
        
        self.assertEqual(evaluated["status"], "success")
        self.assertEqual(evaluated["total_results"], 0)
        self.assertEqual(evaluated["precision"], 0.0)
        self.assertEqual(evaluated["recall"], 0.0)
        
    def test_missing_attributes_handling(self):
        """Test handling of missing attributes in results."""
        evaluator = Evaluator(self.config, MockRetrievalPipeline())
        
        results = [
            {
                "image_id": 1,
                "file_path": "test1.jpg",
                "final_score": 0.8,
                "image_similarity": 0.7,
                "caption_similarity": 0.6,
                "attribute_match": 0.5,
                "attributes": {}  # Missing attributes
            }
        ]
        
        expected_attrs = ["red", "white"]
        expected_env = "office"
        
        # Should not throw KeyError
        evaluated = evaluator._evaluate_query_results(
            query_text="test query",
            query_id=1,
            results=results,
            expected_attributes=expected_attrs,
            expected_environment=expected_env
        )
        
        self.assertEqual(evaluated["status"], "success")
        
    def test_invalid_metadata_handling(self):
        """Test handling of invalid metadata."""
        evaluator = Evaluator(self.config, MockRetrievalPipeline())
        
        results = [
            {
                "image_id": 1,
                "file_path": "test1.jpg",
                "final_score": 0.8,
                "image_similarity": 0.7,
                "caption_similarity": 0.6,
                "attribute_match": 0.5,
                "attributes": {
                    "clothing_items": None,  # Invalid type
                    "color_palette": "red",  # Invalid type
                    "environment": 123  # Invalid type
                }
            }
        ]
        
        expected_attrs = ["red"]
        expected_env = "office"
        
        # Should handle gracefully without crashing
        try:
            evaluated = evaluator._evaluate_query_results(
                query_text="test query",
                query_id=1,
                results=results,
                expected_attributes=expected_attrs,
                expected_environment=expected_env
            )
            self.assertEqual(evaluated["status"], "success")
        except Exception as e:
            self.fail(f"Exception raised: {e}")


class MockRetrievalPipeline:
    """Mock retrieval pipeline for testing."""
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Mock search returning sample results."""
        if "raincoat" in query.lower():
            return [
                {
                    "image_id": 101,
                    "file_path": "data/raw/raincoat1.jpg",
                    "final_score": 0.85,
                    "image_similarity": 0.82,
                    "caption_similarity": 0.78,
                    "attribute_match": 0.75,
                    "attributes": {
                        "clothing_items": ["raincoat", "jacket"],
                        "color_palette": ["yellow", "black"],
                        "environment": "street",
                        "style_vibe": "casual"
                    }
                },
                {
                    "image_id": 102,
                    "file_path": "data/raw/raincoat2.jpg",
                    "final_score": 0.75,
                    "image_similarity": 0.70,
                    "caption_similarity": 0.65,
                    "attribute_match": 0.60,
                    "attributes": {
                        "clothing_items": ["raincoat"],
                        "color_palette": ["yellow"],
                        "environment": "park",
                        "style_vibe": "casual"
                    }
                }
            ]
        elif "office" in query.lower():
            return [
                {
                    "image_id": 201,
                    "file_path": "data/raw/office1.jpg",
                    "final_score": 0.90,
                    "image_similarity": 0.88,
                    "caption_similarity": 0.85,
                    "attribute_match": 0.80,
                    "attributes": {
                        "clothing_items": ["shirt", "tie", "blazer"],
                        "color_palette": ["white", "navy"],
                        "environment": "office",
                        "style_vibe": "formal"
                    }
                }
            ]
        else:
            return [
                {
                    "image_id": 301,
                    "file_path": "data/raw/generic1.jpg",
                    "final_score": 0.60,
                    "image_similarity": 0.55,
                    "caption_similarity": 0.50,
                    "attribute_match": 0.45,
                    "attributes": {
                        "clothing_items": ["shirt", "pants"],
                        "color_palette": ["blue", "gray"],
                        "environment": "unknown",
                        "style_vibe": "casual"
                    }
                }
            ]


if __name__ == '__main__':
    unittest.main()