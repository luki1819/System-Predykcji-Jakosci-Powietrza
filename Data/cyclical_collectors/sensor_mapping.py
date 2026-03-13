import requests
import sqlite3
import time
from typing import List, Dict, Optional


class GIOSSensorsSync:
    """
    klasa co synchronizuuje czujniki z api do tych moich w bazie
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.api_base_url = "https://api.gios.gov.pl/pjp-api/v1/rest"
        self.conn = None
        self.cursor = None

    def connect_db(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            print("polączono z bazą danych")
        except Exception as e:
            print(f"błąd w połączeniu z bazą danych: {e}")
            raise

    def close_db(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("koniec połączenia z baza danych")

    def get_stations(self) -> List[Dict]:
        """
        pobiera stacje i zwraca listę słowników
        """
        query = "SELECT id_stacji_GIOŚ FROM stacje WHERE id_stacji_GIOŚ IS NOT NULL"
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        stations = [dict(row) for row in rows]
        print(f"pobrało się {len(stations)} stacji z bazy danych")
        return stations

    def fetch_sensors_from_api(self, station_id: int) -> Optional[List[Dict]]:
        """
        pobiera czujniki z stacji w api GIOŚ po id tej stacji w systemie gioś
        a zwraca listę czujników
        """
        url = f"{self.api_base_url}/station/sensors/{station_id}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            sensors = data.get("Lista stanowisk pomiarowych dla podanej stacji", [])

            if sensors:
                print(f"Stacja z id: {station_id}: pobrało się {len(sensors)} czujników")
            else:
                print(f"stacja z id: {station_id}: nie ma czujników")

            return sensors

        except requests.exceptions.RequestException as e:
            print(f"Nastąpił błąd podczas poibierania info ze stacji: {station_id}: {e}")
            return None

    def get_sensors_from_db(self, station_id: int) -> List[Dict]:
        """
        Pobiera czujniki z tabeli 'sensory' dla danej stacji
        zwraca sensory w postaci listy
        """
        query = """
                SELECT s.id_sensora, s.nazwa_sensora, st.id_stacji_GIOŚ
                FROM sensory s
                         JOIN stacje st ON s.id_stacji = st.id_stacji
                WHERE st.id_stacji_GIOŚ = ?
                """
        self.cursor.execute(query, (station_id,))
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]

    def update_id_sensora(self, sensor_db_id: int, id_sensora_GIOŚ: int) -> bool:
        """
        aktualizuje pole id_stacji_gios danymi z api gios
        """
        query = """
                UPDATE sensory
                SET id_sensora_GIOŚ = ?
                WHERE id_sensora = ?
                """
        try:
            self.cursor.execute(query, (id_sensora_GIOŚ, sensor_db_id))
            return True
        except Exception as e:
            print(f"błąd przy aktualizacji czujnika: {sensor_db_id}: {e}")
            return False

    def match_and_update_sensors(self, station_id: int, api_sensors: List[Dict],
                                 db_sensors: List[Dict]) -> Dict[str, int]:
        """
        dopasowuje czujniki i je aktualizuje
        Zwraca statystyki ile się udało dopasować
        """
        stats = {'dopasowane': 0, 'niedopasowene': 0}

        api_sensors_map = {}
        for sensor in api_sensors:
            wskaznik = sensor.get('Wskaźnik')
            id_sensora = sensor.get('Identyfikator stanowiska')

            if wskaznik and id_sensora:
                api_sensors_map[wskaznik] = id_sensora

        print(f"    Debug: Dostępne wskaźniki z API: {list(api_sensors_map.keys())}")

        for db_sensor in db_sensors:
            sensor_name = db_sensor['nazwa_sensora']

            id_sensora_GIOŚ = api_sensors_map.get(sensor_name)

            if id_sensora_GIOŚ:
                if self.update_id_sensora(db_sensor['id_sensora'], id_sensora_GIOŚ):
                    print(f"    ✓ Zaktualizowano czujnik '{sensor_name}' -> ID GIOŚ: {id_sensora_GIOŚ}")
                    stats['matched'] += 1
                else:
                    stats['not_matched'] += 1
            else:
                print(f"    ⚠ Nie znaleziono dopasowania dla czujnika '{sensor_name}'")
                stats['not_matched'] += 1

        return stats

    def sync_all_sensors(self, delay: float = 0.5):
        """
        Synchronizuje wszystkie czujniki dla wszystkich stacji
        dodaje opóźnienie żeby nie wywaliło api
        """
        print("\n=== Rozpoczęcie synchronizacji czujników GIOŚ ===\n")

        try:
            self.connect_db()
            stations = self.get_stations()

            total_stats = {'matched': 0, 'not_matched': 0, 'stations_processed': 0}

            for i, station in enumerate(stations, 1):
                station_id = station['id_stacji_GIOŚ']
                print(f"\n[{i}/{len(stations)}] Przetwarzanie stacji ID: {station_id}")

                # pobiera z api
                api_sensors = self.fetch_sensors_from_api(station_id)

                if api_sensors:
                    # pobiera z bazy danych
                    db_sensors = self.get_sensors_from_db(station_id)

                    if db_sensors:
                        stats = self.match_and_update_sensors(station_id, api_sensors, db_sensors)
                        total_stats['matched'] += stats['matched']
                        total_stats['not_matched'] += stats['not_matched']
                        total_stats['stations_processed'] += 1
                    else:
                        print(f"brak czujników w bazie dla stacji {station_id}")

                if i < len(stations):
                    time.sleep(delay)

            self.conn.commit()
            print("\nwszystkie zmiany zostały zatwierdzone")

            # Podsumowanie
            print("\nstatystyki synchronizacji")
            print(f"Przetworzonych stacji: {total_stats['stations_processed']}")
            print(f"Dopasowanych czujników: {total_stats['matched']}")
            print(f"Niedopasowanych czujników: {total_stats['not_matched']}")

        except Exception as e:
            print(f"\n✗ Błąd podczas synchronizacji: {e}")
            import traceback
            print(traceback.format_exc())
            if self.conn:
                self.conn.rollback()
                print("Wycofano zmiany")
        finally:
            self.close_db()



if __name__ == "__main__":

    db_path = '../AQPS.db'

    syncer = GIOSSensorsSync(db_path)
    syncer.sync_all_sensors(delay=0.5)