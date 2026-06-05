import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

class ForecastingAPI:
    @staticmethod
    def upload_file(filename: str, content: bytes, content_type: str) -> requests.Response:
        return requests.post(
            f"{BACKEND_URL}/upload",
            files={"file": (filename, content, content_type)},
            timeout=60,
        )

    @staticmethod
    def get_preflight(file_id: str, forecast_horizon: int, date_col: str, value_col: str) -> requests.Response:
        return requests.post(
            f"{BACKEND_URL}/preflight",
            json={
                "file_id": file_id,
                "forecast_horizon": forecast_horizon,
                "date_col": date_col,
                "value_col": value_col,
            },
            timeout=15,
        )

    @staticmethod
    def submit_analysis(payload: dict) -> requests.Response:
        return requests.post(
            f"{BACKEND_URL}/analyze",
            json=payload,
            timeout=30,
        )

    @staticmethod
    def get_job_status(job_id: str) -> requests.Response:
        return requests.get(f"{BACKEND_URL}/jobs/{job_id}", timeout=10)

    @staticmethod
    def send_chat(file_id: str, query: str) -> requests.Response:
        return requests.post(
            f"{BACKEND_URL}/chat",
            json={"file_id": file_id, "query": query},
            timeout=60
        )