STRUKTURA KATALOGÓW
----------------------
Poniżej przedstawiono opis zawartości poszczególnych folderów znajdujących 
się w katalogu głównym. Każdy z wymienionych folderów posiada własny plik 
README ze szczegółowymi opisami zawartości.

Data
    Centrum danych projektu. Katalog ten zawiera:
    - Główny plik bazy danych SQLite (AQPS.db).
    - Skrypty kolektorów służące do cyklicznego pobierania i 
      aktualizacji pomiarów jakości powietrza (z API GIOŚ) oraz danych 
      meteorologicznych (OpenMeteo).
    - Pliki CSV z danymi historycznymi użytymi do trenowania modeli.

hiperparametry_LSTM
    Moduł badawczy poświęcony sieciom neuronowym. Zawiera skrypty 
    przeprowadzające proces optymalizacji hiperparametrów (przy użyciu 
    frameworka Optuna) dla dwóch badanych architektur:
    - LSTM z jedną warstwą.
    - LSTM z dwiema warstwami.

hiperparametry_XGBoost
    Moduł badawczy poświęcony algorytmowi Gradient Boosting. Zawiera kod 
    odpowiedzialny za dobór optymalnych hiperparametrów dla modelu XGBoost, 
    wraz z implementacją inżynierii cech (tworzenie opóźnień czasowych 
    i statystyk kroczących).

trening_modeli_ustalone_hp
    Moduł treningu finalnego. Znajdują się tu skrypty służące do szkolenia 
    ostatecznych wersji modeli (zarówno LSTM, jak i XGBoost) z wykorzystaniem 
    parametrów wyłonionych w etapie optymalizacji. Skrypty te generują 
    także raporty ewaluacyjne na zbiorze testowym.

Prediction_system
    Katalog zawierający kod źródłowy działającej aplikacji końcowej (System 
    Predykcji). W jego skład wchodzą:
    - Scheduler (zarządca procesów cyklicznych).
    - Dashboard (interfejs użytkownika w technologii Streamlit).
    - Moduły backendowe (przetwarzanie danych, wymagane cechy).
    - Wytrenowane modele.