# Time Series Data Forecaster Agent

An intelligent, multi-agent system for time series forecasting that combines statistical models with AI-powered analysis to provide comprehensive forecasting solutions.

## Overview

The Time Series Data Forecaster Agent is a sophisticated application that automates the entire forecasting pipeline using a multi-agent architecture. It ingests time series data, performs comprehensive analysis, selects optimal forecasting models, generates predictions, and creates detailed reports with actionable insights.

### Key Features

- **Multi-Agent Architecture**: Five specialized agents handle different aspects of the forecasting pipeline
- **Multiple Forecasting Models**: ARIMA, SARIMA, Holt-Winters, and EWMA models
- **AI-Powered Analysis**: Statistical analysis and model selection powered by LLMs
- **Comprehensive Reporting**: Detailed reports with visualizations and business insights
- **Interactive UI**: Streamlit-based web interface for easy data upload and analysis
- **Docker Deployment**: Containerized application for easy deployment
- **RAG Integration**: Memory-augmented analysis with ChromaDB vector database

## Architecture

![Agent Architecture](data_forecaster/docs/agent_arch.jpegagent_arch.jpeg)
![Agent Workflow](backend/docs/agent_workflow.jpeg)

The system consists of five specialized agents working in sequence:

1. **Data Validation Agent**: Validates and preprocesses input data
2. **Statistical Analysis Agent**: Performs comprehensive statistical analysis
3. **Model Selection Agent**: Selects the optimal forecasting model using AI reasoning
4. **Forecasting Agent**: Generates forecasts using multiple statistical models
5. **Report Generation Agent**: Creates comprehensive reports with insights

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local development)
- Google API Key (for Gemini models) or Ollama (for local LLMs)

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd data_forecaster
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys
   ```

3. Start the application:
   ```bash
   cd docker
   docker-compose up --build
   ```

4. Access the application:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000

### Local Development

1. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Install frontend dependencies:
   ```bash
   cd frontend
   pip install -r requirements.txt
   ```

3. Start the backend:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

4. Start the frontend:
   ```bash
   cd frontend
   streamlit run app.py
   ```

## Project Structure

```
data_forecaster/
├── backend/                 # FastAPI backend service
│   ├── agents/             # Specialized AI agents
│   ├── core/               # Configuration and logging
│   ├── forecasting/        # Statistical forecasting models
│   ├── prompts/            # LLM prompts for agents
│   ├── rag/                # RAG knowledge base
│   ├── utils/              # Utility functions
│   ├── main.py             # API endpoints
│   └── orchestrator.py     # Pipeline orchestration
├── frontend/               # Streamlit frontend
│   ├── tabs/               # UI components for different views
│   └── app.py              # Main application
├── data/                   # Sample data
├── docker/                 # Docker configuration
└── docs/                   # Documentation
```

## Agents

### 1. Data Validation Agent
- Validates input data format and quality
- Detects missing values, outliers, and inconsistencies
- Prepares data for analysis

### 2. Statistical Analysis Agent
- Performs comprehensive statistical analysis
- Calculates ACF/PACF, trend, seasonality
- Recommends data transformations

### 3. Model Selection Agent
- Evaluates multiple forecasting models
- Selects optimal model based on data characteristics
- Provides reasoning for model selection

### 4. Forecasting Agent
- Implements multiple statistical forecasting models:
  - ARIMA (AutoRegressive Integrated Moving Average)
  - SARIMA (Seasonal ARIMA)
  - Holt-Winters (Triple Exponential Smoothing)
  - EWMA (Exponentially Weighted Moving Average)
- Generates forecasts with confidence intervals

### 5. Report Generation Agent
- Creates comprehensive analysis reports
- Provides business insights and recommendations
- Generates visualizations and charts

## Supported Models

- **ARIMA**: AutoRegressive Integrated Moving Average for non-seasonal data
- **SARIMA**: Seasonal ARIMA for data with seasonal patterns
- **Holt-Winters**: Triple exponential smoothing for trend and seasonality
- **EWMA**: Exponentially Weighted Moving Average for simple forecasting

## API Endpoints

### Backend (http://localhost:8000)

- `GET /health` - Health check
- `POST /upload` - Upload time series data
- `POST /preflight` - Run data quality checks
- `POST /analyze` - Start forecasting analysis
- `GET /jobs/{job_id}` - Get job status and results
- `POST /chat` - Chat with the analysis results

## Configuration

Environment variables can be set in the `.env` file:

```bash
# LLM Configuration
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL=gemini-1.5-flash
USE_OLLAMA=False
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3

# File Upload Settings
MAX_UPLOAD_MB=100
ALLOWED_EXTENSIONS=csv,xlsx

# Storage
CHROMA_PERSIST_DIR=./chroma_db
```

## Testing

Run backend tests:
```bash
cd backend
python -m pytest tests/
```

## Documentation

- [Forecasting Best Practices](docs/forecasting_best_practices.txt)
- [ARIMA Model Guide](docs/arima.txt)
- [SARIMA Model Guide](docs/sarima.txt)
- [Holt-Winters Model Guide](docs/holt_winters.txt)



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Bala Priya C](https://www.freecodecamp.org/news/author/balapriyac/) for the data cleaning methods in the freeCodeCamp post [*How to Clean Time Series Data in Python*](https://www.freecodecamp.org/news/how-to-clean-time-series-data-in-python/) that inspired the `utils.data_cleaning` module

- [Forecasting: Principles and Practice (3rd ed.)](https://otexts.com/fpp3/) by Hyndman & Athanasopoulos — the canonical reference for the ARIMA, SARIMA and exponential-smoothing methodology implemented in the `backend/forecasting` package

- [Diogo Franquinho](https://diogofranquinho.com/notes/econometrics/time-series-analysis.html) for the concise *Time Series Analysis* technical notes covering stationarity, ACF, ARIMA and AIC-based model selection that informed the RAG knowledge base (`backend/rag/docs/`)
As well as his Udemy courses!

Other resources used

- [Statsmodels](https://www.statsmodels.org/) for statistical modeling
- [Pmdarima](https://alkaline-ml.com/pmdarima/) for ARIMA modeling
- [Langchain](https://github.com/langchain-ai/langchain) for LLM integration
- [Streamlit](https://streamlit.io/) for the frontend framework
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
