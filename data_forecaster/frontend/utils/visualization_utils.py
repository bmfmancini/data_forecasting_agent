"""
Visualization utilities for the Time Series Data Forecaster Agent.
Provides helper functions for rendering dynamic visualizations from LLM-generated Plotly configurations.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

logger = logging.getLogger(__name__)


class DynamicVisualizer:
    """A class to render dynamic visualizations from LLM-generated Plotly configurations."""
    
    def __init__(self):
        """Initialize the DynamicVisualizer."""
        pass
    
    def render_from_config(self, config: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Render a Plotly chart from a configuration dictionary.
        
        Args:
            config: Dictionary containing Plotly chart configuration
            key: Optional unique key for Streamlit elements
            
        Returns:
            bool: True if rendering was successful, False otherwise
        """
        try:
            # Validate the configuration
            if not self._validate_config(config):
                st.warning("Invalid visualization configuration.")
                return False
            
            # Create the figure based on the configuration
            fig = self._create_figure(config)
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key=key)
                return True
            else:
                st.warning("Could not generate visualization from configuration.")
                return False
                
        except Exception as e:
            logger.error(f"Error rendering visualization: {str(e)}")
            st.error(f"Error rendering visualization: {str(e)}")
            return False
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the visualization configuration.
        
        Args:
            config: Dictionary containing Plotly chart configuration
            
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Basic validation - check if config is a dictionary
        if not isinstance(config, dict):
            return False
            
        # Check for required keys
        # For now, we'll be permissive and allow various Plotly chart types
        return True
    
    def _create_figure(self, config: Dict[str, Any]) -> Optional[go.Figure]:
        """
        Create a Plotly figure from the configuration.
        
        Args:
            config: Dictionary containing Plotly chart configuration
            
        Returns:
            go.Figure: Plotly figure object or None if creation failed
        """
        try:
            # Handle different types of configurations
            if "data" in config and "layout" in config:
                # Full figure specification
                return go.Figure(config)
            elif "type" in config:
                # Express-style specification
                return self._create_express_figure(config)
            elif isinstance(config, dict) and config:
                # Try to interpret as a figure specification
                return go.Figure(config)
            else:
                logger.warning("Invalid configuration format")
                return None
                
        except Exception as e:
            logger.error(f"Error creating figure: {str(e)}")
            return None
    
    def _create_express_figure(self, config: Dict[str, Any]) -> Optional[go.Figure]:
        """
        Create a Plotly Express-style figure from the configuration.
        
        Args:
            config: Dictionary containing chart type and data
            
        Returns:
            go.Figure: Plotly figure object or None if creation failed
        """
        try:
            chart_type = config.get("type", "").lower()
            data = config.get("data", {})
            layout = config.get("layout", {})
            
            # Create figure based on type
            if chart_type == "line":
                fig = px.line(**data)
            elif chart_type == "bar":
                fig = px.bar(**data)
            elif chart_type == "scatter":
                fig = px.scatter(**data)
            elif chart_type == "histogram":
                fig = px.histogram(**data)
            elif chart_type == "box":
                fig = px.box(**data)
            elif chart_type == "violin":
                fig = px.violin(**data)
            elif chart_type == "heatmap":
                fig = px.density_heatmap(**data) if config.get("density", False) else px.imshow(**data)
            elif chart_type == "area":
                fig = px.area(**data)
            else:
                # Fallback to generic figure creation
                fig = go.Figure(config)
            
            # Apply layout if provided
            if layout:
                fig.update_layout(**layout)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating express figure: {str(e)}")
            return None


def render_visualization_from_llm(viz_data: Dict[str, Any], key: Optional[str] = None) -> bool:
    """
    Render a visualization from LLM-generated data.
    
    Args:
        viz_data: Dictionary containing visualization data from LLM
        key: Optional unique key for Streamlit elements
        
    Returns:
        bool: True if rendering was successful, False otherwise
    """
    visualizer = DynamicVisualizer()
    return visualizer.render_from_config(viz_data, key)


def parse_llm_visualization_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse visualization configuration from LLM response text.
    
    Args:
        response_text: Text response from LLM that may contain visualization configuration
        
    Returns:
        Dict[str, Any]: Parsed visualization configuration or None if not found
    """
    try:
        # Look for JSON configuration in the response
        # This is a simple implementation - in practice, you might want more robust parsing
        
        # Check if the response is already JSON
        if response_text.strip().startswith('{'):
            return json.loads(response_text)
            
        # Look for JSON-like patterns in the text
        import re
        json_pattern = r'\{(?:[^{}]|(?R))*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                config = json.loads(match)
                # Basic validation - check if it looks like a visualization config
                if isinstance(config, dict) and (config.get("data") or config.get("type") or config.get("layout")):
                    return config
            except json.JSONDecodeError:
                continue
                
        return None
        
    except Exception as e:
        logger.error(f"Error parsing LLM visualization response: {str(e)}")
        return None