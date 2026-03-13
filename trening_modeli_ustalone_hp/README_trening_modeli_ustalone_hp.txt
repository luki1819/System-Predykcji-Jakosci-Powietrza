Katalog ten zawiera implementację pozostałych modeli. Skrypty te nie 
przeprowadzają procesu poszukiwania hiperparametrów, lecz wykorzystują 
wartości ustalone w zmiennych `FIXED_PARAMS`. Służą do wytrenowania 
modeli i ich szczegółowej oceny.

ZAWARTOŚĆ PODKATALOGÓW:

A) LSTM (zawierający plik: lstm.py)
   - Realizuje trening sieci neuronowej na podstawie znalezionych parametrów i jednowarstwowej (lepszej) architektury.
   - Generuje pliki wynikowe: `final_model.keras` (wagi modelu) oraz `scalers.pkl` 
     (obiekty skalujące niezbędne do odwrócenia transformacji).
   - pliki te trafiają do tworzonego wg podanego w skrypcie folderu (istniejącego lub nie)

B) XGBOOST (zawierający plik: model_xgboost.py)
   - Realizuje trening modelu gradientowego (Gradient Boosting).
   - Zawiera pełny potok inżynierii cech (tworzenie lagów, średnich ruchomych 3h/24h, 
     różnicowania) wewnątrz klasy.
   - Generuje wykresy miary błędów, krzywe uczenia
   - pliki k

Każdy ze skryptów tworzy własny folder z wynikami (np. `results_lstm_dwutlenek_azotu`, 
`results_xgboost_dwutlenek_azotu`), zawierający kompletny zestaw metryk ewaluacyjnych 
dla zbioru testowego, sam model oraz wykresy.
Te użyte przy pisaniu pracy zostały już dodane. 