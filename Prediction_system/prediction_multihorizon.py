import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from prepare_data_for_models import merge_station_data
from feature_engineering import process_data

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost niedostępny")


class MultiHorizonPredictor:
    def __init__(self, model_folder):
        self.model_folder = Path(model_folder)
        self.model_name = self.model_folder.name  # np. xgboost_no2 potrzebne do predykcji tam w tabeli bd
        self.model_type = "XGBoost"
        self.model = None
        self.config = None
        self.horizons = []
        self.feature_names = []

    def load_model(self):
        model_path = self.model_folder / "model.pkl"

        if not model_path.exists():
            print(f"Brakuje modelu: {model_path}")
            return False

        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Załadowano model: {model_path}")
            return True
        except Exception as e:
            print(f"Model się nie załaddował błąd: {e}")
            return False

    def load_config(self):
        config_path = self.model_folder / "model_config.json"

        if not config_path.exists():
            print(f"Configu brakuje!: {config_path}")
            return False

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

            self.horizons = self.config.get('horizons', [1, 3, 6, 12])
            self.feature_names = self.config.get('feature_names', [])

            print(f"   Załadowano config")
            print(f"   Horyzonty: {self.horizons}")
            print(f"   Liczba cech: {len(self.feature_names)}")
            return True
        except Exception as e:
            print(f"Błąd ładowania config: {e}")
            return False

    def add_lag_features(self, df):
        df = df.copy()

        pollutants = ['dwutlenek azotu', 'pył zawieszony PM10', 'pył zawieszony PM2.5']

        for pollutant in pollutants:
            if pollutant not in df.columns:
                print(f"Brakuje kolumny: {pollutant}")
                continue

            df[f'{pollutant}_lag1'] = df[pollutant].shift(1)
            df[f'{pollutant}_lag3'] = df[pollutant].shift(3)
            df[f'{pollutant}_lag6'] = df[pollutant].shift(6)
            df[f'{pollutant}_lag12'] = df[pollutant].shift(12)
            df[f'{pollutant}_lag24'] = df[pollutant].shift(24)

            df[f'{pollutant}_rolling_min_3h'] = df[pollutant].rolling(window=3, min_periods=1).min()
            df[f'{pollutant}_rolling_max_3h'] = df[pollutant].rolling(window=3, min_periods=1).max()
            df[f'{pollutant}_rolling_mean_24h'] = df[pollutant].rolling(window=24, min_periods=1).mean()
            df[f'{pollutant}_rolling_min_24h'] = df[pollutant].rolling(window=24, min_periods=1).min()
            df[f'{pollutant}_rolling_max_24h'] = df[pollutant].rolling(window=24, min_periods=1).max()

            df[f'{pollutant}_diff_1h'] = df[pollutant].diff(1)
            df[f'{pollutant}_diff_3h'] = df[pollutant].diff(3)

        weather_features = ['wind_speed_10m', 'wind_direction_10m', 'precipitation']
        for feature in weather_features:
            if feature in df.columns:
                df[f'{feature}_lag1'] = df[feature].shift(1)
                df[f'{feature}_lag3'] = df[feature].shift(3)

        weather_rolling = ['temperature_2m', 'relative_humidity_2m', 'precipitation',
                           'wind_speed_10m', 'wind_direction_10m', 'pressure_msl']
        for feature in weather_rolling:
            if feature in df.columns:
                df[f'{feature}_rolling_mean_24h'] = df[feature].rolling(window=24, min_periods=1).mean()

        if 'temperature_2m' in df.columns:
            df['temp_diff_1h'] = df['temperature_2m'].diff(1)
            df['temp_diff_24h'] = df['temperature_2m'].diff(24)

        return df

    def prepare_features(self, df):
        df = self.add_lag_features(df)
        df = df.dropna()

        if df.empty:
            print("Brak danych po usunięciu NaN")
            return None

        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            print(f"Brakujące cechy ({len(missing_features)})")
            for feature in missing_features:
                df[feature] = 0

        X = df[self.feature_names].values[-1:, :]
        return X

    def predict(self, df):
        if self.model is None or self.config is None:
            print("Model lub config nie załadowany")
            return None

        X = self.prepare_features(df)

        if X is None:
            return None

        try:
            if isinstance(self.model, list):
                predictions = [float(model.predict(X)[0]) for model in self.model]
            else:
                pred = self.model.predict(X)
                if pred.ndim == 2:
                    predictions = pred[0].tolist()
                else:
                    predictions = pred.tolist()

            result = {f'{h}h': pred for h, pred in zip(self.horizons, predictions)}

            print("Predykcje:")
            for horizon, value in result.items():
                print(f"   {horizon:6s}: {value:8.2f} µg/m³")

            return result

        except Exception as e:
            print(f"Błąd podczas predykcji: {e}")
            import traceback
            traceback.print_exc()
            return None


def predict_for_station(station_id, hours_back=48, models_dir="models_for_prediction"):
    print("SYSTEM PREDYKCJI JAKOŚCI POWIETRZA")
    print("-------------------------------------------------------------------------------------------------")

    print("Pobieranie danych...")
    df_raw = merge_station_data(station_id=station_id, hours_back=hours_back)

    if df_raw is None:
        print("błąd pobierania danych z bazy danych")
        return None

    print("\nPrzetwarzanie danych...")
    df_processed = process_data(df_raw)
    print(f"Dane gotowe: {df_processed.shape}")

    models_config = {
        'pył zawieszony PM2.5': 'xgboost_pm25',
        'pył zawieszony PM10': 'xgboost_pm10',
        'dwutlenek azotu': 'xgboost_no2'
    }

    results = {}

    for target_name, folder_name in models_config.items():
        print(f"\n{'=' * 70}")
        print(f"Cel predykcji: {target_name}")
        print(f"{'=' * 70}")

        model_folder = Path(models_dir) / folder_name

        if not model_folder.exists():
            print(f"Folder z modeem nie istnieje: {model_folder}")
            continue

        predictor = MultiHorizonPredictor(model_folder)

        if not predictor.load_model() or not predictor.load_config():
            print(f"pomijanie celu {target_name}")
            continue

        predictions = predictor.predict(df_processed)
        if predictions:
            results[target_name] = predictions

    return results


if __name__ == "__main__":
    results = predict_for_station(station_id=1, hours_back=48)

    if results:
        print("\n" + "*" * 70)
        print("podsumowanie wszystkich predykcji")
        print("*" * 70)
        for target, preds in results.items():
            print(f"\n{target}:")
            for horizon, value in preds.items():
                print(f"  {horizon:6s}: {value:8.2f} µg/m³")