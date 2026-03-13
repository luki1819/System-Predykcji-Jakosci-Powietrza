from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging
from prediction_multihorizon import MultiHorizonPredictor
from prepare_data_for_models import merge_station_data
from feature_engineering import process_data

'''
Ten moduł odpowiada za nadzorowanie całego systemu to tutaj cyklicznie są urchamine skrypty pobierania danych pogodowych
i tych o jakości powietrza. tutaj również pobierane są dane do predykcji modeli. Uruchamiane są predykcje o wyznaczonych
godzinach czyli narazie :15 min po pełnej godzinie żeby na api gioś zdążyły się pojawić pomiary. Wyniki
otrzymane z modułu predykcji są tutaj zapisywane do bazy danych wraz z informacjami o uruchomieniach predykcji.
'''''

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AirQualityScheduler:
    def __init__(self, db_path="../Data/AQPS.db", station_id=1):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.station_id = station_id
        self.scheduler = BackgroundScheduler()
        self.predictors = {}
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tu są tworzone w bazie danych 2 nowe tabele

        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS prognozy
                       (
                           id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                           id_stacji               INTEGER  NOT NULL,
                           cel_predykcji           TEXT     NOT NULL,
                           predykcja_utworzona     DATETIME NOT NULL,
                           czas_prognozowany       DATETIME NOT NULL,
                           horyzont_czasowy        TEXT     NOT NULL,
                           nazwa_modelu            TEXT     NOT NULL,
                           wartosc_predykcji       REAL     NOT NULL,
                           czas_danych_wejsciowych DATETIME,
                           utworzono               DATETIME DEFAULT CURRENT_TIMESTAMP
                       )
                       """)

        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS uruchomienia_predykcji
                       (
                           id                       INTEGER PRIMARY KEY AUTOINCREMENT,
                           id_stacji                INTEGER  NOT NULL,
                           czas_uruchomienia        DATETIME NOT NULL,
                           status                   TEXT     NOT NULL,
                           status_pobierania_danych TEXT,
                           status_predykcji         TEXT,
                           komunikat                TEXT,
                           utworzono                DATETIME DEFAULT CURRENT_TIMESTAMP
                       )
                       """)

        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_pred_station_target
                           ON prognozy (id_stacji, cel_predykcji, czas_prognozowany)
                       """)

        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_pred_made_at
                           ON prognozy (predykcja_utworzona DESC)
                       """)

        conn.commit()
        conn.close()
        logger.info("Baza danych zainicjalizowana")

    def run_data_collectors(self):
        logger.info("zbieranie danych")

        try:
            current_dir = Path(__file__).parent
            data_collectors_dir = current_dir / '..' / 'Data' / 'cyclical_collectors'

            air_collector = data_collectors_dir / 'air_quality_cyc_collector.py' #uzupełnia pomiary ale tylko do 3dni wstecz
            weather_collector = data_collectors_dir / 'weather_cyc_collector.py' #uzupełnia pogodę

            logger.info("zbieranie danych o jakości powietrza")
            result_air = subprocess.run(
                [sys.executable, str(air_collector)],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result_air.returncode != 0:
                logger.error(f"X Błąd air_quality_collector: {result_air.stderr}")
                return False

            logger.info("✅ Dane o jakości powietrza pobrane")

            logger.info("Zbieranie danych pogodowych...")
            result_weather = subprocess.run(
                [sys.executable, str(weather_collector)],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result_weather.returncode != 0:
                logger.error(f"X Błąd weather_collector: {result_weather.stderr}")
                return False

            logger.info("✅ Dane pogodowe pobrane")
            return True

        except Exception as e:
            logger.error(f"X Błąd podczas zbierania danych: {e}")
            return False

    def load_models(self, models_dir="models_for_prediction"):
        logger.info("wczytywanie zapisanych modeli modeli")

        models_config = {
            'pył zawieszony PM2.5': 'xgboost_pm25',
            'pył zawieszony PM10': 'xgboost_pm10',
            'dwutlenek azotu': 'xgboost_no2'
        }

        for target, folder_name in models_config.items():
            model_folder = Path(models_dir) / folder_name

            if not model_folder.exists():
                logger.warning(f"Folder nie istnieje: {model_folder}")
                continue

            predictor = MultiHorizonPredictor(model_folder)

            if predictor.load_model() and predictor.load_config():
                self.predictors[target] = predictor
                logger.info(f"✅ {target}")
            else:
                logger.warning(f"Nie załadowano: {target}")

        logger.info(f"Załadowano {len(self.predictors)} modeli")

    def run_prediction_for_timestamp(self, prediction_time, station_id):
        try:
            logger.info("Pobieranie danych")

            # wubiera dane z perspektywy czasu predykcji jak jest np. 11:37 to robi predykcję 11:00 (api_ma_pomiary z opóźnieniem)
            end_time = prediction_time
            start_time = end_time - timedelta(hours=48)
            logger.info(f"Zakres: {start_time.strftime('%H:%M')} → {end_time.strftime('%H:%M')}")

            df_raw = self.get_data_for_timerange(station_id, start_time, end_time)
            if df_raw is None:
                logger.error("Nie udało się pobrać danych")
                return False

            logger.info("Przetwarzanie danych...")
            df_processed = process_data(df_raw)
            data_from_timestamp = df_processed.index.max()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            total_inserted = 0
            for target, predictor in self.predictors.items():
                logger.info(f"Predykcja dla: {target}")
                predictions = predictor.predict(df_processed)

                if predictions is None:
                    logger.warning(f"Brak predykcji dla {target}")
                    continue

                logger.info(f"Otrzymano predykcje: {list(predictions.keys())}")
                # dodawanie prognoz i informacji o uruchomieniu prognozy

                for horizon_str, value in predictions.items():
                    horizon_hours = int(horizon_str.replace('h', ''))
                    target_timestamp = prediction_time + timedelta(hours=horizon_hours)

                    cursor.execute("""
                                   SELECT id
                                   FROM prognozy
                                   WHERE id_stacji = ?
                                     AND cel_predykcji = ?
                                     AND predykcja_utworzona = ?
                                     AND czas_prognozowany = ?
                                     AND horyzont_czasowy = ?
                                   """, (
                                       station_id,
                                       target,
                                       prediction_time.isoformat(),
                                       target_timestamp.isoformat(),
                                       horizon_str
                                   ))

                    if cursor.fetchone():
                        logger.debug(f"    Duplikat: {horizon_str}")
                        continue

                    cursor.execute("""
                                   INSERT INTO prognozy
                                   (id_stacji, cel_predykcji, predykcja_utworzona, czas_prognozowany,
                                    horyzont_czasowy, nazwa_modelu, typ_modelu, wartosc_predykcji,
                                    czas_danych_wejsciowych, utworzono)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                   """, (
                                       station_id,
                                       target,
                                       prediction_time.isoformat(),
                                       target_timestamp.isoformat(),
                                       horizon_str,
                                       predictor.model_name,  # np. xgboost_no2
                                       predictor.model_type,  # XGBoost
                                       float(value),
                                       data_from_timestamp.isoformat() if data_from_timestamp else None,
                                       datetime.now().isoformat()
                                   ))
                    total_inserted += 1
                    logger.debug(f"    Zapisano: {horizon_str} = {value:.2f}")

            conn.commit()
            conn.close()

            if total_inserted > 0:
                logger.info(f"Zapisano {total_inserted} prognoz do bazy")
                return True
            else:
                logger.warning("Nie zapisano żadnych prognoz")
                return False

        except Exception as e:
            logger.error(f"Błąd predykcji: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def get_data_for_timerange(self, station_id, start_time, end_time):
        #pobieranie danych z danego okresu
        import pandas as pd
        conn = sqlite3.connect(self.db_path)

        # informacje o stacji
        station_info = pd.read_sql(
            "SELECT szerokosc, dlugosc FROM stacje WHERE id_stacji = ?",
            conn,
            params=(station_id,)
        )

        if station_info.empty:
            conn.close()
            return None

        # dane o powietrzu
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
            conn.close()
            return None

        df_air['data_pomiaru'] = pd.to_datetime(df_air['data_pomiaru'], format='mixed')
        df_air_pivot = df_air.pivot_table(
            index='data_pomiaru',
            columns='nazwa_sensora',
            values='wartosc',
            aggfunc='mean'
        )

        # najbliższa stacja pogodowa
        from prepare_data_for_models import find_nearest_weather_location
        weather_id = find_nearest_weather_location(
            station_info['szerokosc'].iloc[0],
            station_info['dlugosc'].iloc[0]
        )

        if weather_id is None:
            conn.close()
            return None

        # dane pogodowe
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
            return None

        df_weather['data_pomiaru'] = pd.to_datetime(df_weather['data_pomiaru'], format='mixed')
        df_weather.set_index('data_pomiaru', inplace=True)

        # łączenie danych
        df_merged = pd.concat([df_air_pivot, df_weather], axis=1)
        df_merged.reset_index(inplace=True)
        df_merged.rename(columns={'data_pomiaru': 'datetime'}, inplace=True)

        return df_merged

    def get_last_prediction_time(self, station_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT MAX(czas_uruchomienia)
                       FROM uruchomienia_predykcji
                       WHERE id_stacji = ?
                         AND status = 'success'
                       """, (station_id,))

        result = cursor.fetchone()
        conn.close()

        if result and result[0]:
            return datetime.fromisoformat(result[0])
        return None

    def calculate_missing_hours(self, last_time, current_time, max_days_back=3):
        """
        liczenie brakujących godzin: zwraca listę timestampów dla których brakuje predykcji
        """
        if last_time is None:
            return [current_time]

        missing = []
        start = last_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        end = current_time.replace(minute=0, second=0, microsecond=0)

        # ograniczenie do X dni wstecz
        max_start = end - timedelta(days=max_days_back)
        if start < max_start:
            logger.warning(f"luka jest zaduża (limit api gioś) brakuje: {max_days_back} dni - ograniczenie do  {max_days_back} dni wstecz")
            start = max_start

        current = start
        while current <= end:
            missing.append(current)
            current += timedelta(hours=1)

        return missing if missing else []

    def run_with_gap_filling(self):
        logger.info("=" * 70)
        logger.info(f"🕐 ROZPOCZĘCIE ZADANIA - {datetime.now()}")
        logger.info("=" * 70)

        now = datetime.now()

        # czas predykcji to zawsze pełna godzina przed aktualnym czasem
        # bo jak jest 11:15, to current_time = 11:00
        # a jak jest 11:03, to current_time = 10:00 (za wcześnie na 11:00)
        if now.minute < 5:
            current_time = (now - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            logger.info(f"Za wcześnie ({now.strftime('%H:%M')}) - wcześniejsza godzina")
        else:
            current_time = now.replace(minute=0, second=0, microsecond=0)

        logger.info(f"Czas predykcji: {current_time.strftime('%Y-%m-%d %H:%M')}")

        data_ok = self.run_data_collectors()

        if not data_ok:
            logger.error("X Błąd zbierania danych")
            self.log_run(self.station_id, current_time, 'error',
                         data_status='failed', pred_status='skipped')
            return

        last_time = self.get_last_prediction_time(self.station_id)

        if last_time:
            gap_hours = (current_time - last_time).total_seconds() / 3600
            logger.info(f"Ostatnia predykcja: {last_time.strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"Luka w predykcjach: {gap_hours:.1f} godzin")

            # Jeśli luka < 1 godziny, to znaczy że już była predykcja dla tej godziny
            if gap_hours < 1.0:
                logger.info("predykcja dla tej godziny już istnieje")
                return
        else:
            logger.info("Pierwsza predykcja dla tej stacji")

        missing_times = self.calculate_missing_hours(last_time, current_time, max_days_back=3)

        if not missing_times:
            logger.info("wszystko aktualne brak godzin do uzupełnienia")
            return

        logger.info(f"Do uzupełnienia: {len(missing_times)} predykcji")

        if len(missing_times) > 1:
            logger.info(f"   Od: {missing_times[0].strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"   Do: {missing_times[-1].strftime('%Y-%m-%d %H:%M')}")

        self.load_models()

        if not self.predictors:
            logger.error("X nie załadowano żadnych modeli!")
            return

        success = 0
        for i, pred_time in enumerate(missing_times, 1):
            logger.info(f"\n{'─' * 70}")
            logger.info(f"[{i}/{len(missing_times)}] {pred_time.strftime('%Y-%m-%d %H:%M')}")

            if self.run_prediction_for_timestamp(pred_time, self.station_id):
                success += 1
                self.log_run(self.station_id, pred_time, 'success',
                             data_status='success', pred_status='success')
            else:
                self.log_run(self.station_id, pred_time, 'error',
                             data_status='success', pred_status='failed')

        logger.info(f"\n{'=' * 70}")
        logger.info(f"ZAKOŃCZONE: {success}/{len(missing_times)} predykcji")
        logger.info(f"{'=' * 70}")

    def log_run(self, station_id, run_time, status, data_status='unknown',
                pred_status='unknown', message=""):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
                       INSERT INTO uruchomienia_predykcji
                       (id_stacji, czas_uruchomienia, status, status_pobierania_danych,
                        status_predykcji, komunikat, utworzono)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       """, (station_id, run_time.isoformat(), status,
                             data_status, pred_status, message, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def start(self, run_immediately=True):
        logger.info(f"Uruchamianie schedulera dla stacji {self.station_id}")

        self.scheduler.add_job(
            self.run_with_gap_filling,
            CronTrigger(minute=15),
            id=f'hourly_prediction_{self.station_id}',
            replace_existing=True
        )

        if run_immediately:
            self.run_with_gap_filling()

        self.scheduler.start()
        logger.info("Scheduler uruchomiony - predykcja co godzinę o :15")

        try:
            import time
            while True:
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            logger.info("|| Zatrzymywanie...")
            self.scheduler.shutdown()
            logger.info("Zatrzymano")


if __name__ == "__main__":
    scheduler = AirQualityScheduler(
        db_path="../Data/AQPS.db",
        station_id=1
    )
    scheduler.start(run_immediately=True)