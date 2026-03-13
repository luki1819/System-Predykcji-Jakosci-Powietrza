Katalog zawiera skrypt python służący do wyboru hiperparametrów modelu XGBoost

Główne funkcjonalności skryptu:
  - inżynieria cech - dodatkowo wymagane cechy opóźnione i statystyki (średnie, min, max)
  - Strategia MultiOutput: Zastosowanie wrappera MultiOutputRegressor pozwalającego 
    na obsługę wielu horyzontów prognozy (1h, 3h, 6h, 12h) przez jeden estymator.
  - Optymalizacja hiperparametrów: Dobór parametrów strukturalnych (głębokość drzewa, 
    wagi dzieci) i uczących (learning rate, gamma) przy użyciu Optuna.

WYNIKI:
Skrypt generuje folder z wynikami zawierający:
  - Wykresy ważności cech (top features) dla każdego horyzontu.
  - Pliki CSV z pełnym rankingiem istotności zmiennych.
  - Wykresy diagnostyczne i metryki błędów (analogicznie do modeli LSTM).
  - Konfigurację najlepszego modelu w formacie JSON oraz sam model.


Pozostały folder - finalny_model to folder zawierający właśnie wyniki uruchomienia skryptu xgboost_optuna.py
