import pandas as pd
import sqlite3
from datetime import datetime
import os

# === KONFIGURACJA ==========================================
DB_PATH = "../AQPS.db"
OUTPUT_FOLDER = "merged_data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# ===========================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


def get_stations_with_weather():
    """pbieranie stacje wraz z ich przypisanymi lokalizacjami pogodowymi"""
    conn = get_connection()

    # pobieranie wszystkie stacje z pomiarami
    query_stations = """
                     SELECT DISTINCT st.id_stacji, \
                                     st.nazwa, \
                                     st.szerokosc, \
                                     st.dlugosc
                     FROM stacje st
                              JOIN sensory s ON st.id_stacji = s.id_stacji
                              JOIN pomiary p ON s.id_sensora = p.sensor_id \
                     """
    df_stations = pd.read_sql(query_stations, conn)

    # pobieranie lokalizacje pogodowe
    query_weather_locs = """
                         SELECT id_lokalizacji, \
                                nazwa, \
                                szerokosc, \
                                dlugosc
                         FROM pogoda_lokalizacje \
                         """
    df_weather_locs = pd.read_sql(query_weather_locs, conn)

    conn.close()

    # przypisz lokalizacje pogodowe do stacji
    stations_mapping = []

    for _, station in df_stations.iterrows():
        # Szukaj najbliższej lokalizacji pogodowej
        best_match = None
        min_distance = float('inf')

        for _, weather_loc in df_weather_locs.iterrows():
            # Oblicz różnicę współrzędnych (przybliżona odległość)
            distance = abs(station['szerokosc'] - weather_loc['szerokosc']) + \
                       abs(station['dlugosc'] - weather_loc['dlugosc'])

            if distance < min_distance:
                min_distance = distance
                best_match = weather_loc

        if best_match is not None:
            stations_mapping.append({
                'id_stacji': station['id_stacji'],
                'nazwa_stacji': station['nazwa'],
                'id_lokalizacji': best_match['id_lokalizacji'],
                'nazwa_lokalizacji': best_match['nazwa'],
                'odleglosc': min_distance
            })

    return pd.DataFrame(stations_mapping)


def get_air_quality_data(id_stacji):
    """Pobiera dane jakości powietrza dla danej stacji"""
    conn = get_connection()

    query = """
            SELECT p.data_pomiaru, \
                   s.nazwa_sensora, \
                   s.jednostka, \
                   p.wartosc
            FROM pomiary p
                     JOIN sensory s ON p.sensor_id = s.id_sensora
            WHERE s.id_stacji = ?
            ORDER BY p.data_pomiaru \
            """

    df = pd.read_sql(query, conn, params=(id_stacji,))
    conn.close()

    # konwersja daty
    df['data_pomiaru'] = pd.to_datetime(df['data_pomiaru'], errors='coerce')

    # Pivot - każdy sensor jako kolumna
    df_pivot = df.pivot_table(
        index='data_pomiaru',
        columns='nazwa_sensora',
        values='wartosc',
        aggfunc='mean'
    )

    return df_pivot


def get_weather_data(id_lokalizacji):
    """Pobiera dane pogodowe dla danej lokalizacji"""
    conn = get_connection()

    query = """
            SELECT data_pomiaru, \
                   temperature_2m, \
                   relative_humidity_2m, \
                   precipitation, \
                   wind_speed_10m, \
                   wind_direction_10m, \
                   pressure_msl, \
                   boundary_layer_height
            FROM pogoda_pomiary
            WHERE id_lokalizacji = ?
            ORDER BY data_pomiaru \
            """

    df = pd.read_sql(query, conn, params=(id_lokalizacji,))
    conn.close()

    # Konwersja daty
    df['data_pomiaru'] = pd.to_datetime(df['data_pomiaru'], errors='coerce')
    df.set_index('data_pomiaru', inplace=True)

    return df


def merge_station_data(id_stacji, id_lokalizacji, nazwa_stacji):
    """Łączy dane jakości powietrza z danymi pogodowymi dla jednej stacji"""
    print(f"\n{'=' * 60}")
    print(f"Przetwarzanie stacji: {nazwa_stacji}")
    print('=' * 60)

    # Pobierz dane
    print(" Pobieranie danych jakości powietrza...")
    df_air = get_air_quality_data(id_stacji)

    print(" Pobieranie danych pogodowych...")
    df_weather = get_weather_data(id_lokalizacji)

    if df_air.empty:
        print(f"  Brak danych jakości powietrza dla stacji {nazwa_stacji}")
        return None

    if df_weather.empty:
        print(f"  Brak danych pogodowych dla lokalizacji")
        return None

    # Określ zakres dat (od pierwszego do ostatniego pomiaru powietrza)
    start_date = df_air.index.min()
    end_date = df_air.index.max()

    print(f" Zakres dat: od {start_date} do {end_date}")

    # Utwórz pełną siatkę godzinową
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    print(f" Liczba godzin w zakresie: {len(full_date_range)}")

    # Reindeksuj dane powietrza do pełnej siatki
    df_air_complete = df_air.reindex(full_date_range)

    # Reindeksuj dane pogodowe do pełnej siatki
    df_weather_complete = df_weather.reindex(full_date_range)

    # Połącz dane
    df_merged = pd.concat([df_air_complete, df_weather_complete], axis=1)

    # Resetuj indeks, aby data była kolumną
    df_merged.reset_index(inplace=True)
    df_merged.rename(columns={'index': 'datetime'}, inplace=True)

    # Dodaj kolumny informacyjne
    df_merged.insert(0, 'station_name', nazwa_stacji)
    df_merged.insert(1, 'station_id', id_stacji)

    print(f" Połączono dane:")
    print(f"   - Wiersze: {len(df_merged)}")
    print(f"   - Kolumny: {len(df_merged.columns)}")
    print(f"   - Parametry powietrza: {len(df_air.columns)}")
    print(f"   - Parametry pogodowe: {len(df_weather.columns)}")

    # Statystyki braków
    total_cells = df_merged.shape[0] * df_merged.shape[1]
    missing_cells = df_merged.isnull().sum().sum()
    missing_percent = (missing_cells / total_cells) * 100
    print(f"   - Procent braków danych: {missing_percent:.2f}%")

    return df_merged


def merge_all_stations():
    """Łączy dane dla wszystkich stacji"""
    print("\n" + "=" * 60)
    print("Próba złączenia danych powietrza i pogody")
    print("=" * 60)

    # Pobierz mapowanie stacji i lokalizacji pogodowych
    print("\n Wyszukiwanie stacji i odpowiadających im lokalizacji pogodowych")
    stations_mapping = get_stations_with_weather()

    if stations_mapping.empty:
        print(" Nie znaleziono żadnych stacji z danymi!")
        return

    print(f"\n Znaleziono {len(stations_mapping)} stacji:")
    for _, row in stations_mapping.iterrows():
        print(f"   • {row['nazwa_stacji']} → {row['nazwa_lokalizacji']} (odległość: {row['odleglosc']:.4f}°)")

    # Przetwórz każdą stację
    merged_datasets = []

    for _, row in stations_mapping.iterrows():
        df_merged = merge_station_data(
            row['id_stacji'],
            row['id_lokalizacji'],
            row['nazwa_stacji']
        )

        if df_merged is not None:
            merged_datasets.append(df_merged)

            # Zapisz do pliku CSV
            safe_name = row['nazwa_stacji'].replace(",", "").replace(" ", "_").replace("-", "_")
            output_path = os.path.join(OUTPUT_FOLDER, f"merged_{safe_name}.csv")
            df_merged.to_csv(output_path, index=False)
            print(f" Zapisano do: {output_path}")

            output_path_excel = os.path.join(OUTPUT_FOLDER, f"merged_{safe_name}.xlsx")
            try:
                df_merged.to_excel(output_path_excel, index=False)
                print(f" Zapisano do: {output_path_excel}")
            except:
                print(f"️  Nie udało się zapisać do Excel")

    # Podsumowanie
    print("\n" + "=" * 60)
    print("PODSUMOWANIE")
    print("=" * 60)
    print(f" Pomyślnie przetworzono: {len(merged_datasets)} stacji")
    print(f" Pliki zapisane w folderze: {os.path.abspath(OUTPUT_FOLDER)}")

    if merged_datasets:
        print("\n Statystyki:")
        for i, df in enumerate(merged_datasets):
            station_name = df['station_name'].iloc[0]
            print(f"\n   {i + 1}. {station_name}")
            print(f"      - Zakres dat: {df['datetime'].min()} → {df['datetime'].max()}")
            print(f"      - Liczba rekordów: {len(df)}")
            print(f"      - Liczba kolumn: {len(df.columns)}")

            # Lista parametrów powietrza
            air_params = [col for col in df.columns if col not in [
                'station_name', 'station_id', 'datetime',
                'temperature_2m', 'relative_humidity_2m', 'precipitation',
                'wind_speed_10m', 'wind_direction_10m', 'pressure_msl',
                'boundary_layer_height'
            ]]
            print(f"      - Parametry powietrza: {', '.join(air_params)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    merge_all_stations()