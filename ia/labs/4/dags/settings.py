import json
import logging
import os
from datetime import timedelta
from typing import Dict, Tuple

LOG_DIR = os.getenv("WEATHER_LOG_DIR", "/opt/airflow/logs")

STAGING_DIR = os.path.join(
    LOG_DIR, os.getenv("WEATHER_STAGING_DIR", "weather_staging")
)

DATA_PATH = os.path.join(
    LOG_DIR, os.getenv("WEATHER_DATA_FILENAME", "weather.csv")
)
VISUALIZATION_PATH = os.path.join(
    LOG_DIR, os.getenv("WEATHER_VIS_DIR", "visualizations")
)
MODEL_RESULTS_PATH = os.path.join(
    LOG_DIR, os.getenv("WEATHER_MODEL_RESULTS_FILENAME", "model_results.json")
)
METRICS_PATH = os.path.join(
    LOG_DIR, os.getenv("WEATHER_METRICS_FILENAME", "metrics.json")
)

DEFAULT_CITIES: Dict[str, Tuple[float, float]] = {
    "vdk": (43.1332, 131.9),
}


def _load_cities() -> Dict[str, Tuple[float, float]]:
    raw = os.getenv("WEATHER_CITIES")
    if raw:
        try:
            data = json.loads(raw)
            return {
                name: (coords[0], coords[1]) for name, coords in data.items()
            }
        except Exception:
            pass
    return DEFAULT_CITIES


CITIES = _load_cities()

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": int(os.getenv("WEATHER_RETRIES", "0")),
    "retry_delay": timedelta(
        seconds=int(os.getenv("WEATHER_RETRY_DELAY_SEC", "60"))
    ),
}


def get_logger(name: str = "weather_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        os.makedirs(LOG_DIR, exist_ok=True)
        handler = logging.FileHandler(os.path.join(LOG_DIR, f"{name}.log"))
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
