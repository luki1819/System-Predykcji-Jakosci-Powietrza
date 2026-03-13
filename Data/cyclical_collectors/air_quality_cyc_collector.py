import requests
import sqlite3
import time
from typing import List, Dict, Optional
from datetime import datetime


class GIOSMeasurementsSync:
    """
    Klasa synchronizująca pomiary z API GIOŚ do bazy danych AQPS
    """

    def __init__(self, db_path: str, page_size: Optional[int] = None):
        """
        Inicjalizacja połączenia z bazą danych

        Args:
            db_path: Ścieżka do pliku bazy danych SQLite
            page_size: Liczba wyników zwracanych z API
        """
        self.db_path = db_path
        self.api_base_url = "https://api.gios.gov.pl/pjp-api/v1/rest"
        self.page_size = page_size
        self.conn = None
        self.cursor = None

    def normalize_datetime(self, date_str: str) -> str:
        """
        normalizacja daty
        """
        date_str = str(date_str).strip()

        # rożne formaty dat
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%d.%m.%Y %H:%M:%S',
            '%d.%m.%Y %H:%M',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d %H:%M',
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                # sekundy zawsze na 00
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue

        print(f"Nie można sparsować daty: {date_str}")
        return date_str

    def connect_db(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            print("Połączono z bazą danych")
        except Exception as e:
            print(f"Błąd połączenia z bazą danych: {e}")
            raise

    def close_db(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("Zamknięto połączenie z bazą danych")

    def get_sensors_with_gios_id(self) -> List[Dict]:
        """
        pobiera info o czujnikach które są w bd
        """
        query = """
                SELECT id_sensora, nazwa_sensora, id_sensora_GIOŚ
                FROM sensory
                WHERE id_sensora_GIOŚ IS NOT NULL
                """
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        sensors = [dict(row) for row in rows]
        print(f"✓ Pobrano {len(sensors)} czujników z bazy danych")
        return sensors

    def fetch_measurements_from_api(self, sensor_gios_id: int) -> Optional[List[Dict]]:
        """
        pobiera pomiary z api
        """
        url = f"{self.api_base_url}/data/getData/{sensor_gios_id}"

        if self.page_size is not None:
            url += f"?size={self.page_size}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            measurements = data.get("Lista danych pomiarowych", [])

            if measurements:
                print(f"  ✓ Czujnik {sensor_gios_id}: pobrano {len(measurements)} pomiarów")
                return measurements
            else:
                print(f"  ⚠ Czujnik {sensor_gios_id}: brak danych")
                return None

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Błąd pobierania pomiarów dla czujnika {sensor_gios_id}: {e}")
            return None

    def insert_measurement(self, id_sensora: int, data_pomiaru: str, wartosc: float) -> bool:
        """
        Wstawia pomiar do tabeli pomiary
        """
        query = """
                INSERT INTO pomiary (id_sensora, data_pomiaru, wartosc, dodano)
                VALUES (?, ?, ?, ?)
                """
        try:
            dodano = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            data_pomiaru_normalized = self.normalize_datetime(data_pomiaru)
            self.cursor.execute(query, (id_sensora, data_pomiaru_normalized, wartosc, dodano))
            return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            print(f"    ✗ Błąd wstawiania pomiaru: {e}")
            return False

    def sync_sensor_measurements(self, sensor: Dict) -> Dict[str, int]:
        """
        Synchronizuje pomiary dla pojedynczego czujnika
        """
        stats = {'inserted': 0, 'skipped': 0, 'errors': 0}

        id_sensora = sensor['id_sensora']
        sensor_gios_id = sensor['id_sensora_GIOŚ']
        sensor_name = sensor['nazwa_sensora']

        print(f"\n  Przetwarzanie czujnika: {sensor_name} (ID GIOŚ: {sensor_gios_id})")

        # Pobierz pomiary z API
        measurements = self.fetch_measurements_from_api(sensor_gios_id)

        if not measurements:
            return stats

        # Przetwarzanie pomiarów
        for measurement in measurements:
            date_str = measurement.get('Data')
            value = measurement.get('Wartość')

            # Pomiń pomiary bez daty lub wartości
            if not date_str or value is None:
                stats['skipped'] += 1
                continue

            # Wstaw pomiar - duplikaty zostaną automatycznie pominięte
            if self.insert_measurement(id_sensora, date_str, value):
                stats['inserted'] += 1
            else:
                stats['skipped'] += 1

        if stats['inserted'] > 0:
            print(f"    ✓ Dodano {stats['inserted']} nowych pomiarów")
        if stats['skipped'] > 0:
            print(f"    ⚠ Pominięto {stats['skipped']} pomiarów (duplikaty lub błędne dane)")

        return stats

    def sync_all_measurements(self, delay: float = 0.5):
        """
        Synchronizuje pomiary dla wszystkich czujników

        Args:
            delay: Opóźnienie między requestami do API (w sekundach)
        """
        print("\n=== Rozpoczęcie synchronizacji pomiarów GIOŚ ===\n")
        if self.page_size is not None:
            print(f"Parametry: size={self.page_size}, delay={delay}s\n")
        else:
            print(f"Parametry: delay={delay}s\n")

        try:
            self.connect_db()
            sensors = self.get_sensors_with_gios_id()

            total_stats = {
                'sensors_processed': 0,
                'total_inserted': 0,
                'total_skipped': 0,
                'total_errors': 0
            }

            for i, sensor in enumerate(sensors, 1):
                print(f"\n[{i}/{len(sensors)}] Czujnik: {sensor['nazwa_sensora']}")

                # Synchronizuj pomiary dla czujnika
                stats = self.sync_sensor_measurements(sensor)

                total_stats['sensors_processed'] += 1
                total_stats['total_inserted'] += stats['inserted']
                total_stats['total_skipped'] += stats['skipped']
                total_stats['total_errors'] += stats['errors']

                self.conn.commit()

                if i < len(sensors):
                    time.sleep(delay)

            # Podsumowanie
            print("\n=== Podsumowanie synchronizacji ===")
            print(f"Przetworzonych czujników: {total_stats['sensors_processed']}")
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

    syncer = GIOSMeasurementsSync(db_path, 800)
    syncer.sync_all_measurements(delay=1)