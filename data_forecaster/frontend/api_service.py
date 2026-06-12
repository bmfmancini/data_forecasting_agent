"""
API service module for the Time Series Data Forecaster Agent.
Provides methods to communicate with the backend forecasting service.
"""

import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# API timeout constants
UPLOAD_TIMEOUT = 60
PREFLIGHT_TIMEOUT = 15
ANALYSIS_TIMEOUT = 30
JOB_STATUS_TIMEOUT = 10
CHAT_TIMEOUT = 60

class ForecastingAPI:
    """API client for the forecasting backend service."""
    
    @staticmethod
    def upload_file(filename: str, content: bytes, content_type: str) -> requests.Response:
        """
        Upload a file to the backend service.
        
        Args:
            filename: Name of the file to upload
            content: File content as bytes
            content_type: MIME type of the file
            
        Returns:
            requests.Response: API response
        """
        return requests.post(
            f"{BACKEND_URL}/upload",
            files={"file": (filename, content, content_type)},
            timeout=UPLOAD_TIMEOUT,
        )

    @staticmethod
    def get_preflight(file_id: str, forecast_horizon: int, date_col: str, value_col: str) -> requests.Response:
        """
        Get preflight information for a file.
        
        Args:
            file_id: ID of the uploaded file
            forecast_horizon: Number of periods to forecast
            date_col: Name of the date column
            value_col: Name of the value column
            
        Returns:
            requests.Response: API response
        """
        return requests.post(
            f"{BACKEND_URL}/preflight",
            json={
                "file_id": file_id,
                "forecast_horizon": forecast_horizon,
                "date_col": date_col,
                "value_col": value_col,
            },
            timeout=PREFLIGHT_TIMEOUT,
        )

    @staticmethod
    def submit_analysis(payload: dict) -> requests.Response:
        """
        Submit analysis request to the backend.
        
        Args:
            payload: Analysis request payload
            
        Returns:
            requests.Response: API response
        """
        return requests.post(
            f"{BACKEND_URL}/analyze",
            json=payload,
            timeout=30,
        )

    @staticmethod
    def get_job_status(job_id: str) -> requests.Response:
        """
        Get the status of a job.
        
        Args:
            job_id: ID of the job to check
            
        Returns:
            requests.Response: API response
        """
        return requests.get(f"{BACKEND_URL}/jobs/{job_id}", timeout=JOB_STATUS_TIMEOUT)

    @staticmethod
    def send_chat(file_id: str | None, query: str) -> requests.Response:
        """
        Send a chat query to the backend.
        
        Args:
            file_id: ID of the file to chat about (optional)
            query: User's chat query
            
        Returns:
            requests.Response: API response
        """
        payload = {"query": query}
        if file_id:
            payload["file_id"] = file_id
            
        return requests.post(
            f"{BACKEND_URL}/chat",
            json=payload,
            timeout=CHAT_TIMEOUT
        )