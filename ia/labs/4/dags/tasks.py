import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
from airflow.exceptions import AirflowSkipException
from airflow.sdk import task
from airflow.sdk.bases.hook import BaseHook
from settings import (
    DATA_PATH,
    METRICS_PATH,
    MODEL_RESULTS_PATH,
    STAGING_DIR,
    VISUALIZATION_PATH,
    get_logger,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

LOGGER = get_logger()


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")


def _write_json(obj: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp)
    return path


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


@task()
def skip_if_disabled(env_var: str, reason: str) -> None:
    if _env_flag(env_var):
        LOGGER.info("skip requested: %s", reason)
        raise AirflowSkipException(reason)


@task()
def fetch_weather(city: str, lat: float, lon: float) -> str:
    LOGGER.info("start: get info")
    conn = BaseHook.get_connection("openweather-api")
    api_key = conn.extra_dejson.get("api-key")

    resp = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"lat": lat, "lon": lon, "appid": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    raw = resp.json()
    path = os.path.join(STAGING_DIR, f"{city}_raw_{_timestamp()}.json")
    _write_json(raw, path)
    LOGGER.info("end: get info -> %s", path)
    return path


@task()
def parse_data(city: str, raw_path: str) -> str:
    LOGGER.info("start: parse data")
    raw = _read_json(raw_path)
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
    path = os.path.join(STAGING_DIR, f"{city}_parsed_{_timestamp()}.json")
    _write_json(data, path)
    LOGGER.info("end: parse data -> %s", path)
    return path


@task()
def prep_data(city: str, parsed_path: str) -> str:
    LOGGER.info("start: prep")
    parsed = _read_json(parsed_path)
    for key in ("temp", "temp_min", "temp_max", "feels_like"):
        if parsed.get(key) is not None:
            parsed[key] = round(parsed[key] - 273.15, 2)

    for key in ("wind_speed",):
        if parsed.get(key) is not None:
            parsed[key] = round(parsed[key], 2)

    dt = datetime.fromtimestamp(parsed.pop("dt"))
    parsed.update(
        {"year": dt.year, "month": dt.month, "day": dt.day, "hour": dt.hour}
    )
    path = os.path.join(STAGING_DIR, f"{city}_prepped_{_timestamp()}.json")
    _write_json(parsed, path)
    LOGGER.info("end: prep -> %s", path)
    return path


@task()
def write_data(prepped_path: str) -> str:
    LOGGER.info("start: data save")
    row = _read_json(prepped_path)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(DATA_PATH):
        df.to_csv(DATA_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(DATA_PATH, index=False)
    LOGGER.info("end: data save - updated df saved to: %s", DATA_PATH)
    return DATA_PATH


@task()
def run_model(csv_path: str) -> str:
    LOGGER.info("start: model training")
    if not os.path.exists(csv_path):
        raise AirflowSkipException("data file not found")
    df = pd.read_csv(csv_path)

    features = ["month", "hour", "pressure", "humidity", "wind_speed"]
    target = "feels_like"

    for col in features + [target]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=features + [target])
    if df.empty:
        raise AirflowSkipException("no usable numeric data for model training")

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {"y_true": y_test.tolist(), "y_pred": y_pred.tolist()}
    with open(MODEL_RESULTS_PATH, "w", encoding="utf-8") as fp:
        json.dump(results, fp)

    LOGGER.info(
        "end: model training -> %s (rows=%d)",
        MODEL_RESULTS_PATH,
        len(df),
    )
    return MODEL_RESULTS_PATH


@task()
def evaluate_model(results_path: str) -> str:
    LOGGER.info("start: model evaluation")
    with open(results_path, "r", encoding="utf-8") as fp:
        results = json.load(fp)
    mae = mean_absolute_error(results["y_true"], results["y_pred"])
    with open(METRICS_PATH, "w", encoding="utf-8") as fp:
        json.dump({"mae": mae}, fp)
    LOGGER.info("model MAE: %.3f -> %s", mae, METRICS_PATH)
    return METRICS_PATH


@task()
def visualize_results(csv_path: str, results_path: str) -> str:
    LOGGER.info("start: visualization")
    os.makedirs(VISUALIZATION_PATH, exist_ok=True)

    with open(results_path, "r", encoding="utf-8") as fp:
        results = json.load(fp)
    y_true = results["y_true"]
    y_pred = results["y_pred"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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

    LOGGER.info("end: visualization saved to: %s", viz_path)
    return viz_path
