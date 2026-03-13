
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

# === KONFIGURACJA ===
DB_PATH = "../Data/AQPS.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def get_available_stations():
    """wyświetla stacjie z pomiarami"""
    conn = get_connection()

    query = """
            SELECT DISTINCT st.id_stacji, st.nazwa, st.szerokosc, st.dlugosc
            FROM stacje st
                     JOIN sensory s ON st.id_stacji = s.id_stacji
                     JOIN pomiary p ON s.id_sensora = p.id_sensora
            ORDER BY st.nazwa \
            """

    df_stations = pd.read_sql(query, conn)
    conn.close()

    print("\nDostępne stacje:")
    for idx, row in df_stations.iterrows():
        print(f"{row['id_stacji']:3d}. {row['nazwa']}")

    return df_stations


def find_nearest_weather_location(station_lat, station_lon):
    """znajduje najbliższą lokalizację pogody dla stacji pogodowej, bo po co pobierać różne prognozy w obrębie Wrocławia"""
    conn = get_connection()

    query = """
            SELECT id_lokalizacji, nazwa, szerokosc, dlugosc
            FROM pogoda_lokalizacje \
            """

    df_weather_locs = pd.read_sql(query, conn)
    conn.close()

    if df_weather_locs.empty:
        return None

    min_distance = float('inf')
    best_match = None

    for _, loc in df_weather_locs.iterrows():
        distance = abs(station_lat - loc['szerokosc']) + abs(station_lon - loc['dlugosc'])
        if distance < min_distance:
            min_distance = distance
            best_match = loc

    return best_match['id_lokalizacji']


def merge_station_data(station_id, hours_back=24):
    """
    łączy dane pomiarowe jakości powietrza z tymi o pogodzie żeby było wszystko w recordach
    """
    conn = get_connection()

    # Pobierz informacje o stacji
    station_info = pd.read_sql(
        "SELECT nazwa, szerokosc, dlugosc FROM stacje WHERE id_stacji = ?",
        conn,
        params=(station_id,)
    )

    if station_info.empty:
        print(f"Nie znaleziono stacji o ID: {station_id}")
        conn.close()
        return None

    station_lat = station_info['szerokosc'].iloc[0]
    station_lon = station_info['dlugosc'].iloc[0]

    # Znajdź najbliższą lokalizację pogodową
    weather_id = find_nearest_weather_location(station_lat, station_lon)
    if weather_id is None:
        print("Nie znaleziono lokalizacji pogodowej")
        conn.close()
        return None

    # Oblicz zakres dat
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours_back)

    # pobieranie danych o jakosci powietrza
    query_air = """
                SELECT p.data_pomiaru, s.nazwa_sensora, p.wartosc
                FROM pomiary p
                         JOIN sensory s ON p.id_sensora = s.id_sensora
                WHERE s.id_stacji = ?
                  AND p.data_pomiaru >= ?
                  AND p.data_pomiaru <= ?
                ORDER BY p.data_pomiaru \
                """

    df_air = pd.read_sql(query_air, conn, params=(station_id, start_time, end_time))

    if df_air.empty:
        print("Brak danych jakości powietrza")
        conn.close()
        return None

    df_air['data_pomiaru'] = pd.to_datetime(df_air['data_pomiaru'], format='mixed')
    df_air_pivot = df_air.pivot_table(
        index='data_pomiaru',
        columns='nazwa_sensora',
        values='wartosc',
        aggfunc='mean'
    )

    # pobierz dane meteorologiczne
    query_weather = """
                    SELECT data_pomiaru,
                           temperature_2m,
                           relative_humidity_2m,
                           precipitation,
                           wind_speed_10m,
                           wind_direction_10m,
                           pressure_msl,
                           boundary_layer_height
                    FROM pogoda_pomiary
                    WHERE id_lokalizacji = ?
                      AND data_pomiaru >= ?
                      AND data_pomiaru <= ?
                    ORDER BY data_pomiaru \
                    """

    df_weather = pd.read_sql(query_weather, conn, params=(weather_id, start_time, end_time))
    conn.close()

    if df_weather.empty:
        print("Brak danych pogodowych")
        return None

    df_weather['data_pomiaru'] = pd.to_datetime(df_weather['data_pomiaru'], format='mixed')
    df_weather.set_index('data_pomiaru', inplace=True)

    # pivotowanie danych aby były połączone w rekordach
    df_merged = pd.concat([df_air_pivot, df_weather], axis=1)
    df_merged.reset_index(inplace=True)
    df_merged.rename(columns={'data_pomiaru': 'datetime'}, inplace=True)
    print(df_merged.head())
    return df_merged



if __name__ == "__main__":
    get_available_stations()
    df = merge_station_data(station_id=1, hours_back=24)
    if df is not None:
        print(f"\nPobrano {len(df)} wierszy, {len(df.columns)} kolumn")
        print(df.head())
    else:
        print("Nie udało się pobrać danych")