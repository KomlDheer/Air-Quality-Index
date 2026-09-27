import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_URL = "https://raw.githubusercontent.com/adityarc19/aqi-india/refs/heads/main/city_day.csv"
MODEL_DIR = "models"


def load_data() -> pd.DataFrame:
    logger.info("Loading AQI dataset...")
    df = pd.read_csv(DATA_URL)
    df["Date"] = pd.to_datetime(df["Date"])

    aqi_map = {
        "Good": 0,
        "Satisfactory": 1,
        "Moderate": 2,
        "Poor": 3,
        "Very Poor": 4,
        "Severe": 5,
    }
    df["AQI_Bucket"] = df["AQI_Bucket"].map(aqi_map)

    logger.info("Dataset shape: %s", df.shape)
    return df


def get_cities(df: pd.DataFrame) -> list:
    cities = [city for city in df["City"].unique() if len(df[df["City"] == city]) > 1600]
    for city in ["Mumbai", "Ahmedabad"]:
        if city in cities:
            cities.remove(city)
    logger.info("Selected cities: %s", cities)
    return cities


def get_separate_df(df: pd.DataFrame, cities: list) -> dict:
    df_list = {}
    for city in cities:
        curr_df = df[df["City"] == city].copy()
        curr_df.set_index("Date", inplace=True)
        full_range = pd.date_range(start=curr_df.index.min(), end=curr_df.index.max(), freq="D")
        curr_df = curr_df.reindex(full_range)

        numerical_cols = [
            col for col in curr_df.columns if curr_df[col].dtype == np.float64 and col != "AQI_Bucket"
        ]
        for col in numerical_cols:
            curr_df[col] = curr_df[col].interpolate(method="linear")
            curr_df[col] = curr_df[col].clip(lower=0)

        curr_df["AQI_Bucket"] = curr_df["AQI_Bucket"].ffill().bfill()
        curr_df = curr_df.ffill().bfill()
        curr_df = curr_df.drop(columns=["City"], errors="ignore")
        df_list[city] = curr_df

    return df_list


def add_time_features(df_list: dict) -> dict:
    for city, curr_df in df_list.items():
        curr_df["Year"] = curr_df.index.year
        curr_df["Month"] = curr_df.index.month
        curr_df["Day"] = curr_df.index.day
        curr_df["DayOfWeek"] = curr_df.index.dayofweek
        curr_df["DayOfYear"] = curr_df.index.dayofyear
        curr_df["WeekOfYear"] = curr_df.index.isocalendar().week.astype(int)
        curr_df["Quarter"] = curr_df.index.quarter
        df_list[city] = curr_df
    return df_list


def get_exog_features(df_list: dict, cities: list) -> dict:
    exog_features = {}
    for city in cities:
        aqi_corr = df_list[city].corr(numeric_only=True)["AQI"]
        aqi_corr_sorted = aqi_corr.sort_values(ascending=False)
        aqi_corr_clean = aqi_corr_sorted.drop(labels=["AQI", "AQI_Bucket"], errors="ignore")
        exog = aqi_corr_clean[(aqi_corr_clean >= 0.4) | (aqi_corr_clean <= -0.3)]
        exog = exog.drop(labels=["Year"], errors="ignore")
        exog_features[city] = exog.index.tolist()
    return exog_features


def build_arima_models(df_list: dict, cities: list, exog_features: dict) -> dict:
    split_date = pd.Timestamp("2020-01-01")
    arima_models = {}
    arima_mae = {}

    for city in cities:
        data = df_list[city]
        train_data = data.loc[data.index < split_date].copy()
        test_data = data.loc[data.index >= split_date].copy()

        selected_exog = [col for col in exog_features[city] if col in train_data.columns]

        adf_stat, p_val = adfuller(data["AQI"])
        logger.info("City: %s | ADF statistic: %.4f | p-value: %.4f", city, adf_stat, p_val)

        try:
            auto_model = pm.auto_arima(
                data["AQI"],
                start_p=1,
                start_q=1,
                max_p=5,
                max_q=5,
                d=None,
                seasonal=False,
                trace=False,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )
            p, d, q = auto_model.order
        except Exception:
            p, d, q = 1, 1, 1

        exog_train = train_data[selected_exog] if selected_exog else None
        model = ARIMA(train_data["AQI"], order=(p, d, q), exog=exog_train)
        fitted_model = model.fit()
        arima_models[city] = fitted_model

        if len(test_data) > 0:
            forecast_start = test_data.index.min()
            forecast_end = test_data.index.max()
            exog_test = test_data.loc[forecast_start:forecast_end, selected_exog] if selected_exog else None
            predictions = fitted_model.predict(start=forecast_start, end=forecast_end, exog=exog_test)
            actual = test_data["AQI"].loc[forecast_start:forecast_end]
            arima_mae[city] = mean_absolute_error(actual, predictions)
            logger.info("ARIMA MAE for %s: %.4f", city, arima_mae[city])

    return {"models": arima_models, "mae": arima_mae}


def build_prophet_models(df_list: dict, cities: list, exog_features: dict) -> dict:
    split_date = pd.Timestamp("2020-01-01")
    prophet_models = {}
    prophet_mae = {}

    for city in cities:
        prophet_df = df_list[city].reset_index().rename(columns={"Date": "ds", "AQI": "y"})
        regressor_cols = [col for col in exog_features[city] if col in prophet_df.columns]

        m = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            holidays_prior_scale=10,
        )

        for col in regressor_cols:
            m.add_regressor(col)

        train_df = prophet_df[prophet_df["ds"] <= split_date][["ds", "y"] + regressor_cols]
        m.fit(train_df)
        prophet_models[city] = m

        future_df = prophet_df[prophet_df["ds"] > split_date][["ds"] + regressor_cols]
        if future_df.empty:
            prophet_mae[city] = np.nan
            continue

        predictions = m.predict(future_df)
        actual = prophet_df[prophet_df["ds"] > split_date]["y"].reset_index(drop=True)
        prophet_mae[city] = mean_absolute_error(actual, predictions["yhat"].reset_index(drop=True))
        logger.info("Prophet MAE for %s: %.4f", city, prophet_mae[city])

    return {"models": prophet_models, "mae": prophet_mae}


def save_models(model_bundle: dict, model_type: str, manager: ModelManager) -> None:
    for city, model in model_bundle["models"].items():
        metadata = {
            "city": city,
            "model_type": model_type,
            "mae": model_bundle["mae"].get(city),
            "saved_at": pd.Timestamp.utcnow().isoformat(),
        }
        manager.save_model(model, f"{model_type}_{city}", metadata=metadata)
        logger.info("Saved %s model for %s", model_type, city)


def main() -> None:
    manager = ModelManager(MODEL_DIR)

    df = load_data()
    cities = get_cities(df)
    df_list = get_separate_df(df, cities)
    df_list = add_time_features(df_list)
    exog_features = get_exog_features(df_list, cities)

    arima_bundle = build_arima_models(df_list, cities, exog_features)
    save_models(arima_bundle, "arima", manager)

    prophet_bundle = build_prophet_models(df_list, cities, exog_features)
    save_models(prophet_bundle, "prophet", manager)

    logger.info("Training complete. Model files stored in %s", MODEL_DIR)


if __name__ == "__main__":
    main()
