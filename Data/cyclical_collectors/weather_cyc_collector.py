import requests
import sqlite3
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta


class OpenMeteoWeatherSync:
    """
    Klasa synchronizująca dane pogodowe z API OpenMeteo do bazy danych AQPS
    """

    def __init__(self, db_path: str):
        """
        Inicjalizacja połączenia z bazą danych

        Args:
            db_path: Ścieżka do pliku bazy danych SQLite
        """
        self.db_path = db_path
        self.api_base_url = "https://api.open-meteo.com/v1/forecast"
        self.conn = None
        self.cursor = None

    def connect_db(self):
        """Nawiązuje połączenie z bazą danych"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            print("✓ Połączono z bazą danych")
        except Exception as e:
            print(f"✗ Błąd połączenia z bazą danych: {e}")
            raise

    def close_db(self):
        """Zamyka połączenie z bazą danych"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("✓ Zamknięto połączenie z bazą danych")

    def get_all_locations(self) -> List[Dict]:
        """
        Pobiera wszystkie lokalizacje z tabeli pogoda_lokalizacje

        zwraca liste słowników z danymi lokalizacji
        """
        query = """
                SELECT id_lokalizacji, nazwa, szerokosc, dlugosc
                FROM pogoda_lokalizacje
                """
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        locations = [dict(row) for row in rows]
        print(f"Pobrano {len(locations)} lokalizacji z bazy danych")
        return locations

    def get_latest_measurement_date(self, location_id: int) -> Optional[str]:
        """
        pobiera date najnowszego pomiaru
        """
        query = """
                SELECT MAX(data_pomiaru) as latest_date
                FROM pogoda_pomiary
                WHERE id_lokalizacji = ?
                """
        self.cursor.execute(query, (location_id,))
        row = self.cursor.fetchone()

        if row and row['latest_date']:
            return row['latest_date']
        return None

    def fetch_weather_from_api(
            self,
            latitude: float,
            longitude: float,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> Optional[Dict]:

        params = {
            'latitude': latitude,
            'longitude': longitude,
            'hourly': [
                'temperature_2m',
                'relative_humidity_2m',
                'precipitation',
                'wind_speed_10m',
                'wind_direction_10m',
                'pressure_msl',
                'boundary_layer_height'
            ],
            'timezone': 'Europe/Warsaw'
        }

        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date

        try:
            response = requests.get(self.api_base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'hourly' in data and data['hourly']:
                time_count = len(data['hourly'].get('time', []))
                print(f"Pobrano {time_count} pomiarów pogodowych")
                return data
            else:
                print(f"Brak danych pogodowych")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Błąd pobierania danych pogodowych: {e}")
            return None

    def insert_measurements_batch(
            self,
            location_id: int,
            measurements: List[Tuple]
    ) -> Tuple[int, int]:
        """
        wstawaia pomiary do bd
        """
        query = """
                INSERT \
                OR IGNORE INTO pogoda_pomiary 
                (id_lokalizacji, data_pomiaru, temperature_2m, relative_humidity_2m, 
                 precipitation, wind_speed_10m, wind_direction_10m, pressure_msl, 
                 boundary_layer_height, dodano)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

        try:
            self.cursor.executemany(query, measurements)
            inserted = self.cursor.rowcount
            skipped = len(measurements) - inserted
            return inserted, skipped
        except Exception as e:
            print(f"    ✗ Błąd wstawiania pomiarów: {e}")
            return 0, len(measurements)

    def sync_location_weather(
            self,
            location: Dict,
            days_back: int = 1
    ) -> Dict[str, int]:

        stats = {'inserted': 0, 'skipped': 0, 'errors': 0}

        location_id = location['id_lokalizacji']
        location_name = location['nazwa']
        latitude = location['szerokosc']
        longitude = location['dlugosc']

        print(f"\n  przetwarzanie lokalizacji: {location_name} ({latitude}, {longitude})")

        # Pobierz datę ostatniego pomiaru
        latest_date = self.get_latest_measurement_date(location_id)

        # Określ zakres dat - końcowa data to dzisiaj
        end_date = datetime.now().strftime('%Y-%m-%d')
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)

        if latest_date:
            print(f"    ostatni pomiar w bazie był: {latest_date}")
            # zaczynaj pobieranie od tego ostatniego pomiaru
            start_date_obj = datetime.fromisoformat(latest_date.replace(' ', 'T'))
            start_date = (start_date_obj + timedelta(hours=1)).strftime('%Y-%m-%d')
        else:
            # Brak pomiarów - pobierz ostatnie N dni
            start_date_obj = datetime.now() - timedelta(days=days_back)
            start_date = start_date_obj.strftime('%Y-%m-%d')
            print(f"    Brak pomiarów - pobieranie od: {start_date}")


        weather_data = self.fetch_weather_from_api(
            latitude,
            longitude,
            start_date,
            end_date
        )

        if not weather_data or 'hourly' not in weather_data:
            return stats


        hourly = weather_data['hourly']
        times = hourly.get('time', [])

        if not times:
            print(f"nie ma timestampów w odpowiedzi API")
            return stats

        measurements_list = []
        dodano = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_time = datetime.now().replace(minute=0, second=0, microsecond=0)

        for i, time_str in enumerate(times):
            # zmiana formatu daty żeby była spójna z bd
            measurement_time = datetime.fromisoformat(time_str)
            data_pomiaru = measurement_time.strftime('%Y-%m-%d %H:%M:%S')

            # pomijanie prognoz bo pod tym endpointem są
            if measurement_time > current_time:
                continue

            if latest_date and data_pomiaru <= latest_date:
                stats['skipped'] += 1
                continue

            measurement = (
                location_id,
                data_pomiaru,
                hourly.get('temperature_2m', [])[i] if i < len(hourly.get('temperature_2m', [])) else None,
                hourly.get('relative_humidity_2m', [])[i] if i < len(hourly.get('relative_humidity_2m', [])) else None,
                hourly.get('precipitation', [])[i] if i < len(hourly.get('precipitation', [])) else None,
                hourly.get('wind_speed_10m', [])[i] if i < len(hourly.get('wind_speed_10m', [])) else None,
                hourly.get('wind_direction_10m', [])[i] if i < len(hourly.get('wind_direction_10m', [])) else None,
                hourly.get('pressure_msl', [])[i] if i < len(hourly.get('pressure_msl', [])) else None,
                hourly.get('boundary_layer_height', [])[i] if i < len(
                    hourly.get('boundary_layer_height', [])) else None,
                dodano
            )
            measurements_list.append(measurement)

        if measurements_list:
            # Wstaw pomiary wsadowo
            inserted, skipped = self.insert_measurements_batch(location_id, measurements_list)
            stats['inserted'] = inserted
            stats['skipped'] += skipped

            if inserted > 0:
                print(f"Dodano {inserted} nowych pomiarów")
            if stats['skipped'] > 0:
                print(f"Pominięto {stats['skipped']} pomiarów")
        else:
            print(f"Brak nowych pomiarów do dodania")

        return stats

    def sync_all_weather(self, days_back: int = 1, delay: float = 1.0):
        """
        Synchronizuje dane pogodowe dla wszystkich lokalizacji
        """
        print("\n rozpoczęcie synchronizacji danych pogodowych OpenMeteo\n")
        print(f"Parametry: days_back={days_back}, delay={delay}s")
        print(f"Pobieranie danych do aktualnej godziny (bez prognoz)\n")

        try:
            self.connect_db()
            locations = self.get_all_locations()

            if not locations:
                print("Brak lokalizacji do synchronizacji")
                return

            total_stats = {
                'locations_processed': 0,
                'total_inserted': 0,
                'total_skipped': 0,
                'total_errors': 0
            }

            for i, location in enumerate(locations, 1):
                print(f"\n[{i}/{len(locations)}] Lokalizacja: {location['nazwa']}")

                # Synchronizuj dane pogodowe dla lokalizacji
                stats = self.sync_location_weather(location, days_back)

                total_stats['locations_processed'] += 1
                total_stats['total_inserted'] += stats['inserted']
                total_stats['total_skipped'] += stats['skipped']
                total_stats['total_errors'] += stats['errors']


                self.conn.commit()

                # opóźnienie między requestami
                if i < len(locations):
                    time.sleep(delay)

            # Podsumowanie
            print("\n=== Podsumowanie synchronizacji ===")
            print(f"Przetworzonych lokalizacji: {total_stats['locations_processed']}")
            print(f"Dodanych pomiarów: {total_stats['total_inserted']}")
            print(f"Pominiętych pomiarów: {total_stats['total_skipped']}")
            print(f"Błędów: {total_stats['total_errors']}")

        except Exception as e:
            print(f"\n✗ Błąd podczas synchronizacji: {e}")
            import traceback
            print(traceback.format_exc())
            if self.conn:
                self.conn.rollback()
                print("✗ Wycofano zmiany")
        finally:
            self.close_db()



if __name__ == "__main__":

    db_path = '../Data/AQPS.db'
    syncer = OpenMeteoWeatherSync(db_path)
    syncer.sync_all_weather()
