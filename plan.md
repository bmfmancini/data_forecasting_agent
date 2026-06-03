# Plan: Data Forecaster

## Decisions
- LLM: Groq API (Llama 3.3 70B)
- Time series: Univariate only (1 date col + 1 numeric col)
- Deployment: Local Docker (docker-compose)
- Agents: LangChain ReAct AgentExecutor with Python tool functions
- Charts: Plotly figures serialised as JSON, returned in API response; Matplotlib ACF/PACF as base64 PNG
- Analysis API: synchronous (POST /analyze returns full results when complete)

## Project Structure
```
data_forecaster/
├── backend/
│   ├── main.py                        # FastAPI: /upload, /analyze, /health
│   ├── orchestrator.py                # Coordinates all 5 agents in sequence
│   ├── schemas.py                     # Pydantic I/O models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                  # Settings (env vars, file size limits, allowed types)
│   │   └── logging_config.py          # Python logging setup (file + console handlers)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── data_validation_agent.py
│   │   ├── statistical_analysis_agent.py
│   │   ├── model_selection_agent.py
│   │   ├── forecasting_agent.py
│   │   └── report_generation_agent.py
│   ├── forecasting/
│   │   ├── __init__.py
│   │   ├── holt_winters.py
│   │   ├── arima_model.py
│   │   └── sarima_model.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── knowledge_base.py          # ChromaDB + Sentence Transformers
│   │   └── docs/                      # Raw methodology .txt files
│   │       ├── arima.txt
│   │       ├── sarima.txt
│   │       ├── holt_winters.txt
│   │       ├── forecasting_best_practices.txt
│   │       └── statistical_interpretation.txt
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_parser.py             # CSV/XLSX → DataFrame + column detection
│   │   ├── visualization.py           # Plotly/Matplotlib chart builders
│   │   └── statistical.py             # ADF/KPSS/STL/ACF helper functions
│   └── requirements.txt
├── frontend/
│   ├── app.py                         # Streamlit UI
│   └── requirements.txt
├── data/
│   └── sample_airline_passengers.csv  # Public sample dataset for demo/testing
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── docs/
│   ├── architecture.md                # System architecture diagram (Mermaid) + component overview
│   ├── agent_workflow.md              # Agent pipeline diagram + data flow
│   ├── user_guide.md
│   ├── deployment_guide.md
│   └── security_considerations.md    # Input validation, guardrails, assumptions, limitations
├── logs/                              # Runtime log output (gitignored)
├── .env.example
└── .gitignore
```

## Phases & Steps

### Phase 1 — Project Scaffold ✅
1. ✅ Create full directory tree (including `backend/core/`, `data/`, `logs/`)
2. ✅ `.gitignore`: Python cache, `.env`, `logs/`, `chroma_db/`, `__pycache__`, `.venv`, temp upload dir
3. ✅ `backend/requirements.txt`: fastapi, uvicorn, langchain, langchain-groq, langchain-community, chromadb, sentence-transformers, statsmodels, pandas, numpy, scipy, plotly, matplotlib, python-multipart, openpyxl, python-dotenv, pmdarima
4. ✅ `frontend/requirements.txt`: streamlit, requests, plotly, pandas
5. ✅ `.env.example`: `GROQ_API_KEY=`, `MAX_UPLOAD_MB=10`, `ALLOWED_EXTENSIONS=csv,xlsx`
6. ✅ `backend/core/config.py`: Settings loaded from `.env`; exposes `MAX_UPLOAD_BYTES`, `ALLOWED_MIME_TYPES`, `GROQ_API_KEY`, `CHROMA_PERSIST_DIR`
7. ✅ `backend/core/logging_config.py`: Python `logging` with `RotatingFileHandler` → `logs/app.log` + `StreamHandler`; all modules call `get_logger(__name__)`
8. ✅ Docker files: Dockerfile.backend, Dockerfile.frontend, docker-compose.yml (two services, ports 8000/8501, `logs/` and `chroma_db/` volume mounts)
9. ✅ `data/sample_airline_passengers.csv`: classic public dataset for demo/testing

### Phase 2 — Data Layer ✅
10. ✅ `backend/schemas.py`: UploadResponse, AnalyzeRequest, ValidationResult, StatisticalResult, ModelSelectionResult, ForecastResult, AnalysisResponse (all Pydantic)
11. ✅ `backend/utils/data_parser.py`: parse_upload() reads CSV/XLSX → pd.DataFrame, auto-detects date + value column, parses dates, sorts, returns cleaned DataFrame and detected column names

### Phase 3 — RAG Knowledge Base ✅
12. ✅ Write 5 methodology documents in `backend/rag/docs/` (ARIMA, SARIMA, Holt-Winters, best practices, statistical interpretation — ~300-500 words each)
13. ✅ `backend/rag/knowledge_base.py`: RAGKnowledgeBase class using ChromaDB (persist_directory=./chroma_db), all-MiniLM-L6-v2 sentence-transformers embeddings, load_documents() chunks and upserts on startup, retrieve(query, k=3) → list[str]

### Phase 4 — Forecasting Models ✅
14. ✅ `backend/forecasting/holt_winters.py`: fit_holt_winters(series, forecast_horizon) → {forecast, lower_ci, upper_ci, rmse, mae, mape}
15. ✅ `backend/forecasting/arima_model.py`: fit_arima(series, forecast_horizon) using pmdarima auto_arima → same output schema
16. ✅ `backend/forecasting/sarima_model.py`: fit_sarima(series, forecast_horizon, seasonal_period) using pmdarima auto_arima(seasonal=True) → same output schema

### Phase 5 — Statistical Analysis Utilities ✅
17. ✅ `backend/utils/statistical.py`: helper functions used as agent tools:
    - run_adf_test(series) → {statistic, p_value, is_stationary, interpretation}
    - run_kpss_test(series) → {statistic, p_value, is_stationary, interpretation}
    - run_stl_decomposition(series, period) → {trend, seasonal, residual} as lists
    - compute_acf_pacf(series, lags) → {acf_values, pacf_values, lags}
    - run_periodogram(series) → {dominant_period, frequencies, power}
    - detect_trend(series) → {has_trend, slope, interpretation}

### Phase 6 — Agents ✅
Each agent: LangChain AgentExecutor (ReAct) + Groq LLM + @tool decorated functions

18. ✅ `data_validation_agent.py`: tools = check_missing_timestamps, check_duplicates, check_missing_values, check_irregular_intervals, detect_frequency, assess_size → returns ValidationResult
19. ✅ `statistical_analysis_agent.py`: tools = run_adf_test, run_kpss_test, run_stl_decomposition, compute_acf_pacf, run_periodogram, detect_trend → returns StatisticalResult
20. ✅ `model_selection_agent.py`: tools = evaluate_holt_winters_suitability, evaluate_arima_suitability, evaluate_sarima_suitability → reasons over statistical results, returns ModelSelectionResult with chosen model + explanation + why others rejected
21. ✅ `forecasting_agent.py`: tools = run_holt_winters, run_arima, run_sarima, compute_all_metrics → fits selected model, returns ForecastResult
22. ✅ `report_generation_agent.py`: tools = retrieve_from_rag (calls knowledge_base.retrieve()), generate_section → writes 6-section report using RAG context, returns full report string

### Phase 7 — Orchestrator ✅
23. ✅ `backend/orchestrator.py`: run_pipeline(df, forecast_horizon) executes agents 1→2→3→4→5 sequentially, each receiving prior results; returns AnalysisResponse

### Phase 8 — Visualizations ✅
24. ✅ `backend/utils/visualization.py`:
    - plot_historical(series) → Plotly JSON
    - plot_stl(trend, seasonal, residual) → Plotly JSON (4-panel subplot)
    - plot_acf_pacf(acf, pacf, lags) → Matplotlib base64 PNG
    - plot_forecast(historical, forecast, lower_ci, upper_ci) → Plotly JSON
    - plot_model_comparison(metrics_dict) → Plotly JSON bar chart

### Phase 9 — FastAPI Backend (with Safety Controls) ✅
25. ✅ `backend/main.py`:
    - POST /upload: validate MIME type (csv/xlsx only) + file size (≤ MAX_UPLOAD_BYTES from config) before processing — HTTP 400 with descriptive message on violation; parse → store df in UUID-keyed in-memory dict → return `UploadResponse`
    - POST /analyze: `{file_id, forecast_horizon, date_col?, value_col?}` → run_pipeline() + all 5 charts → AnalysisResponse; HTTP 404 if file_id not found
    - GET /health: returns {status: ok}
    - CORS enabled for Streamlit origin
    - All endpoints log via `logging_config` (request entry, key decisions, errors)

### Phase 10 — Streamlit Frontend ✅
26. ✅ `frontend/app.py` layout:
    - Sidebar: file uploader (CSV/XLSX), column mapping dropdowns (auto-populated), forecast horizon slider (7–365 for daily, or proportional), "Run Analysis" button
    - Main area: 6 tabs:
      - Overview: dataset preview table + historical chart
      - Data Quality: validation summary cards
      - Statistical Analysis: ADF/KPSS results, STL chart, ACF/PACF image
      - Model Selection: selected model + rationale text
      - Forecast: forecast chart + metrics table
      - Report: full analyst report rendered as markdown

### Phase 11 — Documentation
27. `docs/architecture.md`: Mermaid system diagram, component overview, data flow
28. `docs/agent_workflow.md`: Mermaid agent pipeline diagram showing data passed between agents
29. `docs/user_guide.md`: step-by-step usage with screenshot descriptions, sample dataset instructions
30. `docs/deployment_guide.md`: prerequisites (Docker, GROQ_API_KEY), virtual environment setup (python -m venv), docker-compose up steps, env var setup
31. `docs/security_considerations.md`: input validation approach, guardrails, assumptions, limitations

### Phase 12 — Submission Packaging ✅
32. ✅ Confirmed `data/sample_airline_passengers.csv` is included (public dataset, 144 rows)
33. ✅ Verified `.gitignore` excludes `.env`, `logs/`, `chroma_db/`, `.venv`

## Verification
1. ✅ `docker compose build` builds both images without errors (backend + frontend, CPU-only torch)
2. Upload `sample_airline_passengers.csv` via Streamlit; confirm file_id returned with correct frequency
3. Run analysis; confirm all 5 agents execute and return non-empty results
4. Verify all 5 Plotly charts render in Streamlit tabs
5. Verify ACF/PACF PNG displays in Statistical Analysis tab
6. Verify RAG report cites forecasting methodology in natural language
7. Confirm Docker volume persists ChromaDB between container restarts
8. Test XLSX upload path
9. ✅ Test oversized file upload → HTTP 400 returned (`File too large`)
10. ✅ Test invalid file type upload → HTTP 400 returned (`Unsupported content type 'text/plain'`)
11. Check `logs/app.log` contains structured log entries for each step

## Excluded Scope
- Multivariate time series
- Cloud deployment
- User authentication / session persistence beyond in-memory dict
- Model auto-hyperparameter grid search beyond pmdarima auto_arima defaults
- Async background jobs (analysis is synchronous)
