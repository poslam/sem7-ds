import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from airflow.models import Variable
from airflow.sdk import dag, task

DATA_PATH = "/Users/poslam/Downloads/projects/fefu/7/ds/ia/prac/28_11_25/data/weather.csv"
CITIES = {
    "vdk": (43.1332, 131.9),
    "khv": (48.4814, 135.0721),
}

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
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
        api_key = Variable.get("OPENWEATHER_API_KEY")
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
        return resp.json()

    @task()
    def parse_data(raw: dict) -> dict:
        return {
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

    @task()
    def prep_data(parsed: dict) -> dict:
        for key in ("temp", "temp_min", "temp_max", "feels_like"):
            if parsed.get(key) is not None:
                parsed[key] = parsed[key] - 273.15

        dt = datetime.fromtimestamp(parsed.pop("dt"))
        parsed.update(
            {
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "hour": dt.hour,
            }
        )
        return parsed

    @task()
    def write_data(row: dict) -> str:
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df = pd.DataFrame([row])
        if os.path.exists(DATA_PATH):
            df.to_csv(DATA_PATH, mode="a", header=False, index=False)
        else:
            df.to_csv(DATA_PATH, index=False)

        return DATA_PATH

    for name, (lat, lon) in CITIES.items():
        raw = fetch_weather(lat, lon)
        parsed = parse_data(raw)
        prepped = prep_data(parsed)
        write_data(prepped)


dag_w = weather_pipeline()
