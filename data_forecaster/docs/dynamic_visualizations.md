# Dynamic Visualization System

The Time Series Data Forecaster Agent now includes a powerful dynamic visualization system that allows users to request any applicable time series visualization through natural language commands.

## Overview

The dynamic visualization system enables the LLM to generate Plotly chart configurations based on user requests, which are then rendered in the frontend. This provides flexibility to create custom visualizations beyond the predefined chart types.

## How It Works

1. **User Request**: User asks for a specific visualization through the chat interface
2. **LLM Processing**: The LLM analyzes the request and generates a Plotly JSON configuration
3. **Configuration Generation**: The configuration includes data, layout, and chart type information
4. **Frontend Rendering**: The frontend receives the configuration and renders the chart using Plotly

## Supported Visualization Types

The system supports any Plotly chart type that is applicable to time series data:

### Statistical Visualizations
- Histograms and density plots
- Box plots and violin plots
- Q-Q plots for normality testing
- Scatter plots for relationship analysis

### Time-Based Visualizations
- Line charts (standard time series plots)
- Area charts
- Heatmaps for seasonal patterns
- Calendar plots for daily data

### Decomposition Visualizations
- Trend component plots
- Seasonal component plots
- Residual component plots

### Forecast-Specific Visualizations
- Confidence intervals
- Forecast error plots
- Residual diagnostic plots
- Fan charts for multiple scenarios

### Correlation Visualizations
- ACF/PACF plots
- Cross-correlation plots
- Correlation matrices/heatmaps

### Anomaly Detection Visualizations
- Control charts
- Anomaly score plots
- Outlier detection visualizations

### Model Comparison Visualizations
- Error metric comparisons
- Forecast comparison overlays
- Model performance dashboards

## Configuration Format

The LLM generates visualization configurations in JSON format compatible with Plotly:

```json
{
  "data": [
    {
      "x": ["2020-01", "2020-02", "2020-03"],
      "y": [100, 120, 110],
      "type": "scatter",
      "mode": "lines+markers"
    }
  ],
  "layout": {
    "title": "Sample Time Series",
    "xaxis": {"title": "Date"},
    "yaxis": {"title": "Value"}
  }
}
```

Or using Express-style configuration:

```json
{
  "type": "line",
  "data": {
    "x": ["2020-01", "2020-02", "2020-03"],
    "y": [100, 120, 110]
  },
  "layout": {
    "title": "Sample Time Series"
  }
}
```

## Usage Examples

Users can request visualizations using natural language:

- "Show me a histogram of the data values"
- "Create a box plot to show outliers"
- "Generate a heatmap of seasonal patterns"
- "Plot the forecast errors over time"
- "Show me a scatter plot of actual vs predicted values"

## Implementation Details

### Backend
- The orchestrator processes chat requests and detects visualization configurations in LLM responses
- JSON configurations are extracted and passed to the frontend through the API
- The system supports both predefined visualization types and dynamic configurations

### Frontend
- The `DynamicVisualizer` class handles rendering of Plotly configurations
- Visualization utilities provide helper functions for parsing and rendering
- Error handling ensures graceful degradation when configurations are invalid

## Security Considerations

The system includes validation to ensure only appropriate chart types are rendered:
- Configurations are validated before rendering
- Only Plotly chart properties are allowed
- Malformed configurations are handled gracefully

## Extending the System

To add new visualization capabilities:
1. Update the LLM prompt to include new chart types
2. Enhance the `DynamicVisualizer` class to support new chart types
3. Add validation rules for new configuration formats
4. Test with various data scenarios

## Troubleshooting

Common issues and solutions:
- **Invalid JSON**: Ensure the LLM generates valid JSON configurations
- **Missing data**: Verify that required data fields are included in the configuration
- **Rendering errors**: Check that chart types and properties are supported by Plotly
