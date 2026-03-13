import sqlite3
import requests
import pandas as pd
from datetime import datetime, date, timedelta
import time


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_tables_if_not_exist()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _create_tables_if_not_exist(self):
        """Tworzy tabele pogodowe, jeśli nie istnieją"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Tabela lokalizacji pogodowych
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS pogoda_lokalizacje
                           (
                               id_lokalizacji
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               nazwa
                               TEXT,
                               szerokosc
                               REAL
                               NOT
                               NULL,
                               dlugosc
                               REAL
                               NOT
                               NULL,
                               UNIQUE
                           (
                               szerokosc,
                               dlugosc
                           )
                               )
                           ''')

            # Tabela pomiarów pogodowych
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS pogoda_pomiary
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               id_lokalizacji
                               INTEGER
                               NOT
                               NULL,
                               data_pomiaru
                               TEXT
                               NOT
                               NULL,
                               temperature_2m
                               REAL,
                               relative_humidity_2m
                               REAL,
                               precipitation
                               REAL,
                               wind_speed_10m
                               REAL,
                               wind_direction_10m
                               REAL,
                               pressure_msl
                               REAL,
                               boundary_layer_height
                               REAL,
                               dodano
                               TEXT
                               NOT
                               NULL,
                               UNIQUE
                           (
                               id_lokalizacji,
                               data_pomiaru
                           ),
                               FOREIGN KEY
                           (
                               id_lokalizacji
                           ) REFERENCES pogoda_lokalizacje
                           (
                               id_lokalizacji
                           )
                               )
                           ''')

            conn.commit()

    def get_or_create_location(self, nazwa: str, szerokosc: float, dlugosc: float) -> int:
        """Pobiera ID lokalizacji lub tworzy nową"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Sprawdź czy lokalizacja już istnieje
            cursor.execute('''
                           SELECT id_lokalizacji
                           FROM pogoda_lokalizacje
                           WHERE szerokosc = ?
                             AND dlugosc = ?
                           ''', (szerokosc, dlugosc))

            result = cursor.fetchone()
            if result:
                return result[0]

            # Jeśli nie istnieje, utwórz nową
            cursor.execute('''
                           INSERT INTO pogoda_lokalizacje (nazwa, szerokosc, dlugosc)
                           VALUES (?, ?, ?)
                           ''', (nazwa, szerokosc, dlugosc))
            conn.commit()
            return cursor.lastrowid

    def get_all_distinct_dates(self, id_lokalizacji: int) -> list[date]:
        """Pobiera wszystkie unikalne daty dla danej lokalizacji"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                               SELECT DISTINCT date (data_pomiaru)
                               FROM pogoda_pomiary
                               WHERE id_lokalizacji = ?
                               ORDER BY date (data_pomiaru)
                               ''', (id_lokalizacji,))
                return [datetime.strptime(row[0], '%Y-%m-%d').date() for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            return []

    def save_data_bulk(self, id_lokalizacji: int, pomiary_lista: list[tuple]):
        """Zapisuje wiele pomiarów pogodowych naraz (bulk insert)"""
        if not pomiary_lista:
            return 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                               INSERT
                               OR IGNORE INTO pogoda_pomiary (
                    id_lokalizacji, data_pomiaru, temperature_2m, 
                    relative_humidity_2m, precipitation, wind_speed_10m,
                    wind_direction_10m, pressure_msl, 
                    boundary_layer_height, dodano
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                               ''', pomiary_lista)
            conn.commit()
            return cursor.rowcount


class OpenMeteoAPIClient:
    OM_API_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, latitude: float, longitude: float, hourly_params: list[str]):
        self.latitude = latitude
        self.longitude = longitude
        self.hourly_params = hourly_params

    def fetch_historical_data(self, start_date: date, end_date: date) -> pd.DataFrame | None:
        """Pobiera dane historyczne dla zadanego okresu"""
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "hourly": ",".join(self.hourly_params)
        }

        try:
            response = requests.get(self.OM_API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'hourly' in data:
                return pd.DataFrame(data['hourly'])
            else:
                print(f'⚠️ API nie zwróciło danych hourly dla okresu od {start_date} do {end_date}.')
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Błąd zapytania do API: {e}")
            return None


class WeatherUpdater:
    def __init__(self, config: dict):
        self.config = config
        self.db_manager = DatabaseManager(config['DB_PATH'])
        self.api_client = OpenMeteoAPIClient(
            config['LATITUDE'],
            config['LONGITUDE'],
            config['HOURLY_PARAMS']
        )
        self.desired_start_date = datetime.strptime(config['DESIRED_START_DATE_STR'], '%Y-%m-%d').date()

        # Pobierz lub utwórz lokalizację
        self.id_lokalizacji = self.db_manager.get_or_create_location(
            config['LOCATION_NAME'],
            config['LATITUDE'],
            config['LONGITUDE']
        )
        print(f"📍 Używam lokalizacji: {config['LOCATION_NAME']} (ID: {self.id_lokalizacji})")

    def _find_gaps(self) -> list[tuple[date, date]]:
        """Znajduje wszystkie brakujące okresy w danych"""
        existing_dates = self.db_manager.get_all_distinct_dates(self.id_lokalizacji)
        today = date.today() - timedelta(days=1)
        gaps = []

        if not existing_dates:
            gaps.append((self.desired_start_date, today))
            return gaps

        db_start_date, db_end_date = existing_dates[0], existing_dates[-1]

        if db_start_date > self.desired_start_date:
            gaps.append((self.desired_start_date, db_start_date - timedelta(days=1)))

        for i in range(len(existing_dates) - 1):
            if (existing_dates[i + 1] - existing_dates[i]).days > 1:
                gaps.append((existing_dates[i] + timedelta(days=1), existing_dates[i + 1] - timedelta(days=1)))

        if db_end_date < today:
            gaps.append((db_end_date + timedelta(days=1), today))

        return gaps

    def run_update(self):
        """Uruchamia główny proces wyszukiwania i uzupełniania braków w danych"""
        print("\n--- Rozpoczynam inteligentną aktualizację danych pogodowych ---")
        gaps_to_fill = self._find_gaps()

        if not gaps_to_fill:
            print("\n✅ Baza danych pogodowych jest w pełni kompletna i aktualna!")
            return

        print(f"\nZnaleziono {len(gaps_to_fill)} okres(ów) do uzupełnienia. Rozpoczynam pobieranie...")

        for start_date, end_date in gaps_to_fill:
            print(f"\n--- Przetwarzam brakujący okres: od {start_date} do {end_date} ---")

            for month_start in pd.date_range(start=start_date, end=end_date, freq='MS'):
                month_end = min(month_start + pd.offsets.MonthEnd(0), pd.to_datetime(end_date)).date()
                print(f"📥 Pobieram dane za: {month_start.strftime('%Y-%m-%d')} do {month_end.strftime('%Y-%m-%d')}")

                df = self.api_client.fetch_historical_data(month_start.date(), month_end)

                if df is not None and not df.empty:
                    # Konwersja DataFrame na listę tupli
                    pomiary_lista = []
                    dodano = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    for _, row in df.iterrows():
                        data_pomiaru = pd.to_datetime(row['time']).strftime('%Y-%m-%d %H:%M:%S')

                        # Pomiń przyszłe pomiary
                        if pd.to_datetime(data_pomiaru) > datetime.utcnow():
                            continue

                        pomiary_lista.append((
                            self.id_lokalizacji,
                            data_pomiaru,
                            row.get('temperature_2m'),
                            row.get('relative_humidity_2m'),
                            row.get('precipitation'),
                            row.get('wind_speed_10m'),
                            row.get('wind_direction_10m'),
                            row.get('pressure_msl'),
                            row.get('boundary_layer_height'),
                            dodano
                        ))

                    if pomiary_lista:
                        dodano_liczba = self.db_manager.save_data_bulk(self.id_lokalizacji, pomiary_lista)
                        print(f"   ✅ Zapisano: {dodano_liczba} nowych pomiarów")
                        pominiete = len(pomiary_lista) - dodano_liczba
                        if pominiete > 0:
                            print(f"   ⏭️  Pominięto: {pominiete} duplikatów")

                time.sleep(1)  # Opóźnienie między requestami

        print("\n✅ Wszystkie braki w danych zostały uzupełnione. Zakończono proces.")


if __name__ == "__main__":
    # --- GŁÓWNA KONFIGURACJA ---
    CONFIG = {
        'DB_PATH': 'AQPS.db',
        'LOCATION_NAME': 'Wrocław - wyb. Conrada-Korzeniowskiego',
        'LATITUDE': 51.129378,
        'LONGITUDE': 17.02925,
        'DESIRED_START_DATE_STR': '2020-01-01',
        'HOURLY_PARAMS': [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "pressure_msl",
            "boundary_layer_height"
        ]
    }

    # Uruchomienie całego procesu
    updater = WeatherUpdater(CONFIG)
    updater.run_update()