Folder Data zawiera wszystkie pliki i skrypty związane z pobieraniem, przetwarzaniem i analizą danych wykorzystywanych w systemie.

folder cyclical_collectors – zawiera skrypty odpowiedzialne za cykliczne aktualizowanie danych o jakości powietrza i danych pogodowych. Zawiera także jednorazowy skrypt przypisujący czujnikom identyfikatory GIOŚ.

folder data_for_model – folder zawierający finalny plik CSV z dodatkowymi cechami wykorzystywanymi podczas trenowania modeli.

data_merge – ma zestaw skryptów służących do łączenia danych pomiarowych z pogodowymi, tworzenia dodatkowych cech oraz analizy charakterystyk danych (np. sezonowości). Zawiera również wykresy generowane podczas analiz.

data_quality –zawiera skrypty badające jakość danych (w szczególności braki danych) oraz foldery z wynikami analiz w formie wykresów.

folder_z_pomiarami – katalog z surowymi plikami CSV pobranymi z GIOŚ. Stanowi źródło danych dla budowy bazy.

AirQualityCollector.py – skrypt wczytujący pliki CSV z folder_z_pomiarami, tworzący bazę danych i uzupełniający ją pomiarami jakości powietrza.

WeatherCollector.py – skrypt pobierający historyczne dane meteorologiczne z API OpenMeteo i zapisujący je do bazy danych.

AQPS.db – baza danych SQLite przechowująca pomiary jakości powietrza, dane pogodowe oraz wyniki predykcji modeli.