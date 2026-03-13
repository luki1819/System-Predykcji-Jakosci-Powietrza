import numpy as np
import pandas as pd


def fill_missing_values(df):
    """
    wypełnia brakujące pomiary jak są
    """
    df = df.copy()

    # interpolacja prosta dla pogody
    weather_cols = [
        'temperature_2m', 'relative_humidity_2m', 'precipitation',
        'wind_speed_10m', 'wind_direction_10m', 'pressure_msl',
        'boundary_layer_height'
    ]

    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')

    # dla danych jakości powietza to forward fill i bfill
    air_quality_cols = [col for col in df.columns if col not in weather_cols + ['datetime']]

    for col in air_quality_cols:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].ffill().bfill()

    return df


def create_time_features(df):
    df = df.copy()

    # jako indeks
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df.index.name = 'czas_pomiaru'

    # te podstawowe temporalne
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year

    # pory roku
    def get_season(month):
        if month in [1, 2, 12]:
            return 3  # zima
        elif month in [3, 4, 5]:
            return 0  # wiosna
        elif month in [6, 7, 8]:
            return 1  # lato
        else:
            return 2  # jesien

    df['season'] = df['month'].apply(get_season)

    #indeksy pomocnicze do mgły i stagancji

    if 'pressure_msl' in df.columns and 'wind_speed_10m' in df.columns:
        df['air_stagnation'] = (df['pressure_msl'] / (df['wind_speed_10m'] + 0.1)).round(2)

    if 'relative_humidity_2m' in df.columns and 'temperature_2m' in df.columns:
        df['fog_potential'] = (df['relative_humidity_2m'] * np.maximum(0, (10 - df['temperature_2m']))).round(2)

    return df


def process_data(df):
    """

    """
    cols_to_drop = ['station_id', 'station_name']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # wypełnij braki
    df = fill_missing_values(df)

    # zrób cechy jak do xgboosta
    df = create_time_features(df)

    if 'boundary_layer_height' in df.columns:
        df = df.drop(columns=['boundary_layer_height'])

    return df



if __name__ == "__main__":
    from prepare_data_for_models import merge_station_data

    # pobierz czyste pomiary dla danej stcji
    df_raw = merge_station_data(station_id=1, hours_back=24)

    if df_raw is not None:

        df_processed = process_data(df_raw)

        print(f"wymiary danych (kształt): {df_processed.shape}")
        print(f"\nKolumny:\n{list(df_processed.columns)}")
        print(f"\nBraki danych:\n{df_processed.isnull().sum().sum()}")
        print(f"\nPierwsze wiersze:\n{df_processed.head()}")