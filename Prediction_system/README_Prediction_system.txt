Katalog Prediction_systen zawiera pliki źródłowe tworzące system informatyczny predykcji jakości pwoietrza (AQPS).

KLUCZOWE KOMPONENTY:
1. scheduler.py – Najważniejszy komponent systemu. Działa w tle, uruchamiając cyklicznie procesy:
   - Pobierania nowych danych (integracja z kolektorami).
   - Przetwarzania danych i inżynierii cech.
   - Generowania prognoz przez modele XGBoost.
   - Archiwizacji wyników w bazie danych.

2. Dashboard.py – Warstwa prezentacji. Aplikacja webowa umożliwiająca:
   - Podgląd aktualnych stężeń PM2.5, PM10, NO2.
   - Analizę prognoz na 1h, 3h, 6h i 12h w przód.
   - Odczyt rekomendacji zdrowotnych opartych na wyliczonym AQI.
   - Porównanie prognoz historycznych z rzeczywistością.

3. start_dashboard.bat – plik uruchomiający interfejs.

PLIKI WSPOMAGAJĄCE:
- prepare_data_for_models.py: Obsługa połączeń z bazą danych i łączenie tabel.
- feature_engineering.py: Algorytmy czyszczenia danych i ekstrakcji cech.
- prediction_multihorizon.py: Logika przeprowadzania predykcji, wczytuje modele i konfiguracje znajdujące się w folderze Models_for_prediction.


FOLDER Models_for_predition to miejsce do którego trafiły wyniki treningu modeli dla poszczególnych zanieczyszczeń.
Katalog podzielony jest na podfoldery dedykowane konkretnym zanieczyszczeniom 
(np. xgboost_pm10, xgboost_no2). W każdym z nich znajdują się:
  - model.pkl: Plik binarny zawierający wytrenowany model XGBoost.
  - model_config.json: Plik tekstowy z metadanymi (lista używanych cech, horyzonty czasowe).
  - pozostałe pliki jak wykresy, krzywe uczenia itp. nie są one używane, a raczej pozwalają na identyfikację modelu i jego ocenę.
