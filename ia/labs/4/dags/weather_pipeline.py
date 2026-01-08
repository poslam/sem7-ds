import logging
import os
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
from airflow.hooks.base import BaseHook
from airflow.sdk import dag, task
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

DATA_PATH = "/opt/airflow/logs/weather.csv"
VISUALIZATION_PATH = "/opt/airflow/logs/visualizations"
CITIES = {
    "vdk": (43.1332, 131.9),
    # "khv": (48.4814, 135.0721),
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
    dag_id="weather_pipeline",
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
        for key in ("temp_min", "temp_max", "feels_like"):
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
    def run_model(csv_path: str) -> dict:
        logger = logging.getLogger(__name__)
        logger.info("start: model training")

        df = pd.read_csv(csv_path)

        features = [
            "month",
            "hour",
            "pressure",
            "humidity",
            "wind_speed",
        ]
        df = df.dropna(subset=features + ["feels_like"])

        X = df[features]
        y = df["feels_like"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        logger.info("end: model training")

        return {
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        }

    @task()
    def evaluate_model(results: dict) -> float:
        logger = logging.getLogger(__name__)
        logger.info("start: model evaluation")

        y_true = results["y_true"]
        y_pred = results["y_pred"]

        mae = mean_absolute_error(y_true, y_pred)

        logger.info("model MAE: %.3f", mae)

        return mae

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

    @task()
    def visualize_results(
        csv_path: str,
        model_results: dict,
    ) -> str:
        logger = logging.getLogger(__name__)
        logger.info("start: visualization")

        os.makedirs(VISUALIZATION_PATH, exist_ok=True)

        # Создание визуализации результатов модели
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # График: Предсказанные vs Реальные значения
        y_true = model_results["y_true"]
        y_pred = model_results["y_pred"]

        axes[0].scatter(y_true, y_pred, alpha=0.6, color="blue")
        axes[0].plot(
            [min(y_true), max(y_true)],
            [min(y_true), max(y_true)],
            "r--",
            lw=2,
            label="Perfect prediction",
        )
        axes[0].set_xlabel("Реальная температура (°C)")
        axes[0].set_ylabel("Предсказанная температура (°C)")
        axes[0].set_title("Предсказания модели vs Реальные значения")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # График: История температуры из CSV
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            axes[1].plot(
                df.index,
                df["temp"],
                marker="o",
                linestyle="-",
                color="green",
                label="Temperature",
            )
            axes[1].set_xlabel("Индекс записи")
            axes[1].set_ylabel("Температура (°C)")
            axes[1].set_title("История температуры")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = os.path.join(VISUALIZATION_PATH, f"results_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(viz_path, dpi=100, bbox_inches="tight")
        plt.close()

        logger.info("end: visualization saved to: %s", viz_path)

        return viz_path

    for name, (lat, lon) in CITIES.items():
        raw = fetch_weather(lat, lon)
        parsed = parse_data(raw)
        prepped = prep_data(parsed)
        path = write_data(prepped)
        model_results = run_model(path)
        evaluate_model(model_results)
        visualize_results(path, model_results)


dag_w = weather_pipeline()
