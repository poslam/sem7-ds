import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from airflow.hooks.base import BaseHook
from airflow.sdk import dag, task

DATA_PATH = "/opt/airflow/logs/weather.csv"
CITIES = {
    "vdk": (43.1332, 131.9),
    "khv": (48.4814, 135.0721),
}

os.makedirs(
    os.path.dirname("/opt/airflow/logs/weather_pipeline.log"), exist_ok=True
)

logger = logging.getLogger("weather_pipeline")
handler = logging.FileHandler("/opt/airflow/logs/weather_pipeline.log")
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.setLevel(logging.DEBUG)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(seconds=1),
}


@dag(
    default_args=default_args,
    schedule="0 * * * *",
    start_date=datetime(2025, 11, 28),
    catchup=False,
    tags=["weather"],
)
def weather_pipeline():
    @task()
    def fetch_weather(lat: float, lon: float) -> dict:
        logger = logging.getLogger(__name__)
        logger.info("start: get info")

        conn = BaseHook.get_connection("openweather-api")
        api_key = conn.extra_dejson.get("api-key")

        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "lat": lat,
                "lon": lon,
                "appid": api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()

        logger.info("end: get info")

        return resp.json()

    @task()
    def parse_data(raw: dict) -> dict:
        logger = logging.getLogger(__name__)
        logger.info("start: parse data")
        try:
            data = {
                "temp": raw["main"].get("temp"),
                "temp_min": raw["main"].get("temp_min"),
                "temp_max": raw["main"].get("temp_max"),
                "feels_like": raw["main"].get("feels_like"),
                "pressure": raw["main"].get("pressure"),
                "humidity": raw["main"].get("humidity"),
                "wind_speed": raw.get("wind", {}).get("speed"),
                "dt": raw.get("dt"),
                "weather": raw.get("weather", [{}])[0].get("main"),
            }
        except Exception as e:  
            logger.error(f"error: parse data - {str(e)}")

        logger.info("end: parse data")

        return data

    @task()
    def prep_data(parsed: dict) -> dict:
        logger = logging.getLogger(__name__)
        logger.info("start: prep")
        for key in ("temp", "temp_min", "temp_max", "feels_like"):
            if parsed.get(key) is not None:
                parsed[key] = round(parsed[key] - 273.15, 2)

        for key in "wind_speed":
            if parsed.get(key) is not None:
                parsed[key] = round(parsed[key], 2)

        dt = datetime.fromtimestamp(parsed.pop("dt"))
        parsed.update(
            {
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "hour": dt.hour,
            }
        )
        logger.info("end: prep")
        return parsed

    @task()
    def write_data(row: dict) -> str:
        logger = logging.getLogger(__name__)
        logger.info("start: data save")

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df = pd.DataFrame([row])
        if os.path.exists(DATA_PATH):
            df.to_csv(DATA_PATH, mode="a", header=False, index=False)
        else:
            df.to_csv(DATA_PATH, index=False)

        logger.info("end: data save - updated df saved to: %s", DATA_PATH)

        return DATA_PATH

    for name, (lat, lon) in CITIES.items():
        raw = fetch_weather(lat, lon)
        parsed = parse_data(raw)
        prepped = prep_data(parsed)
        write_data(prepped)


dag_w = weather_pipeline()
