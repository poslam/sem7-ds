import importlib
from datetime import datetime
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
from airflow.sdk import dag
from settings import CITIES, DATA_PATH, DEFAULT_ARGS, get_logger
from tasks import (
    evaluate_model,
    fetch_weather,
    parse_data,
    prep_data,
    run_model,
    skip_if_disabled,
    visualize_results,
    write_data,
)

TRIGGER_DAGRUN_PATHS = (
    "airflow.providers.standard.operators.trigger_dagrun.TriggerDagRunOperator",
    "airflow.operators.trigger_dagrun.TriggerDagRunOperator",
    "airflow.operators.dagrun_operator.TriggerDagRunOperator",
    "airflow.operators.dagrun_trigger.TriggerDagRunOperator",
)


def _load_trigger_op():
    for path in TRIGGER_DAGRUN_PATHS:
        module_path, attr = path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            return getattr(module, attr)
        except Exception:
            continue
    raise ImportError(
        "TriggerDagRunOperator not found; tried: "
        + ", ".join(TRIGGER_DAGRUN_PATHS)
    )


if TYPE_CHECKING:
    from airflow.operators.trigger_dagrun import (
        TriggerDagRunOperator as _TriggerDagRunOperator,
    )

    TriggerDagRunOperator = _TriggerDagRunOperator
else:
    TriggerDagRunOperator = _load_trigger_op()

LOGGER = get_logger()


@dag(
    dag_id="weather_data_pipeline",
    default_args=DEFAULT_ARGS,
    schedule="0 * * * *",
    start_date=datetime(2025, 11, 28),
    catchup=False,
    tags=["weather"],
)
def weather_data_pipeline():
    guard = skip_if_disabled("WEATHER_SKIP_DATA", "data collection disabled")
    writes = []
    for name, (lat, lon) in CITIES.items():
        raw = fetch_weather.override(task_id=f"fetch_weather_{name}")(
            name, lat, lon
        )
        parsed = parse_data.override(task_id=f"parse_{name}")(name, raw)
        prepped = prep_data.override(task_id=f"prep_{name}")(name, parsed)
        write_task = write_data.override(task_id=f"write_{name}")(prepped)
        guard >> raw
        writes.append(write_task)

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_weather_training",
        trigger_dag_id="weather_training_pipeline",
        reset_dag_run=True,
        wait_for_completion=False,
    )

    if writes:
        writes >> trigger_training
    else:
        guard >> trigger_training


@dag(
    dag_id="weather_training_pipeline",
    default_args=DEFAULT_ARGS,
    schedule=None,
    start_date=datetime(2025, 11, 28),
    catchup=False,
    tags=["weather"],
)
def weather_training_pipeline():
    guard = skip_if_disabled(
        "WEATHER_SKIP_TRAINING", "model training disabled"
    )
    model_results = run_model(DATA_PATH)
    metrics = evaluate_model(model_results)
    visualization = visualize_results(DATA_PATH, model_results)

    guard >> model_results
    model_results >> metrics >> visualization


dag_collect = weather_data_pipeline()
dag_train = weather_training_pipeline()
