"""
Test script for the dynamic visualization system.
"""

from utils.visualization_utils import DynamicVisualizer
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def test_dynamic_visualizer():
    """Test the DynamicVisualizer class with various chart types."""
    print("Testing DynamicVisualizer...")

    visualizer = DynamicVisualizer()

    # Test 1: Simple line chart configuration
    print("\n1. Testing line chart...")
    line_config = {
        "data": {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 11, 12, 13, 14],
            "type": "scatter",
            "mode": "lines+markers",
        },
        "layout": {
            "title": "Test Line Chart",
            "xaxis": {"title": "X Axis"},
            "yaxis": {"title": "Y Axis"},
        },
    }

    success = visualizer.render_from_config(line_config, key="test_line")
    print(f"Line chart rendering: {'SUCCESS' if success else 'FAILED'}")

    # Test 2: Bar chart configuration
    print("\n2. Testing bar chart...")
    bar_config = {
        "data": {"x": ["A", "B", "C", "D"], "y": [10, 15, 13, 17], "type": "bar"},
        "layout": {"title": "Test Bar Chart"},
    }

    success = visualizer.render_from_config(bar_config, key="test_bar")
    print(f"Bar chart rendering: {'SUCCESS' if success else 'FAILED'}")

    # Test 3: Express-style histogram
    print("\n3. Testing express-style histogram...")
    # Create sample data
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)

    hist_config = {
        "type": "histogram",
        "data": {"x": data, "nbins": 30},
        "layout": {"title": "Test Histogram"},
    }

    success = visualizer.render_from_config(hist_config, key="test_hist")
    print(f"Histogram rendering: {'SUCCESS' if success else 'FAILED'}")

    # Test 4: Box plot
    print("\n4. Testing box plot...")
    box_config = {
        "type": "box",
        "data": {"y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20], "name": "Sample Data"},
        "layout": {"title": "Test Box Plot"},
    }

    success = visualizer.render_from_config(box_config, key="test_box")
    print(f"Box plot rendering: {'SUCCESS' if success else 'FAILED'}")

    print("\nAll tests completed.")


if __name__ == "__main__":
    test_dynamic_visualizer()
