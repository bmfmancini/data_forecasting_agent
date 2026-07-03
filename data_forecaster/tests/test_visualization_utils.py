"""
Unit tests for the visualization utilities.
"""

import unittest
from unittest.mock import patch, MagicMock


class TestVisualizationUtils(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock Streamlit to avoid import issues during testing
        sys.modules["streamlit"] = MagicMock()
        sys.modules["plotly"] = MagicMock()
        sys.modules["plotly.graph_objects"] = MagicMock()
        sys.modules["plotly.express"] = MagicMock()

    def test_import_visualization_utils(self):
        """Test that visualization utilities can be imported."""
        try:
            from utils.visualization_utils import (
                DynamicVisualizer,
                render_visualization_from_llm,
                parse_llm_visualization_response,
            )

            self.assertTrue(True, "Visualization utilities imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import visualization utilities: {e}")

    def test_dynamic_visualizer_initialization(self):
        """Test that DynamicVisualizer can be initialized."""
        try:
            from utils.visualization_utils import DynamicVisualizer

            visualizer = DynamicVisualizer()
            self.assertIsInstance(visualizer, DynamicVisualizer)
        except Exception as e:
            self.fail(f"Failed to initialize DynamicVisualizer: {e}")

    def test_parse_llm_visualization_response_with_json(self):
        """Test parsing LLM response with JSON configuration."""
        from utils.visualization_utils import parse_llm_visualization_response

        # Test with valid JSON
        response = (
            '{"data": {"x": [1, 2, 3], "y": [1, 2, 3]}, "layout": {"title": "Test"}}'
        )
        result = parse_llm_visualization_response(response)
        self.assertIsNotNone(result)
        self.assertIn("data", result)
        self.assertIn("layout", result)

    def test_parse_llm_visualization_response_with_embedded_json(self):
        """Test parsing LLM response with embedded JSON configuration."""
        from utils.visualization_utils import parse_llm_visualization_response

        # Test with embedded JSON
        response = 'Here is the visualization configuration: {"data": {"x": [1, 2, 3], "y": [1, 2, 3]}, "layout": {"title": "Test"}}'
        result = parse_llm_visualization_response(response)
        self.assertIsNotNone(result)
        self.assertIn("data", result)
        self.assertIn("layout", result)


if __name__ == "__main__":
    unittest.main()
