Katalog koncentruje się na procesie doboru optymalnych hiperparametrów oraz treningu 
sieci neuronowych typu LSTM. Zawiera dwa główne skrypty różniące się głębokością 
architektury modelu:

1. lstm_optuna_jedna_warstwa.py – model z pojedynczą warstwą rekurencyjną.
2. lstm_optuna_dwie_warstwy.py  – model głęboki z dwiema warstwami rekurencyjnymi.

Główne funkcjonalności obu skryptów:
  - Wczytanie, wstępne przetwarzanie i skalowanie danych.
  - Automatyczna optymalizacja hiperparametrów z wykorzystaniem biblioteki Optuna 
    (przestrzeń przeszukiwań obejmuje: liczbę neuronów, dropout, learning rate, 
    wielkość okna czasowego).
  - Trening finalnego modelu z wykorzystaniem mechanizmu Early Stopping.
  - Ewaluacja modelu na zbiorze testowym dla horyzontów czasowych: 1, 3, 6 i 12 godzin.
  - Generowanie wykresów diagnostycznych i zapis wyników.

STRUKTURA WYNIKÓW:
Pozostałe podkatalogi zawierają rezultaty przeprowadzonych procesów optymalizacji 
i treningu. W ich skład wchodzą:
  - Wykresy (historia uczenia, predykcje, ważność hiperparametrów).
  - Pliki JSON ze szczegółowymi metrykami błędów.
  - Zapisane obiekty scalerów.
  - Zapisane wagi wytrenowanych modeli.