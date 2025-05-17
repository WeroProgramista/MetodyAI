# Porównanie Metod AI (ARIMA vs LSTM) w Prognozowaniu Cen Aktywów Finansowych 📊📈

## Opis Projektu

Projekt ten, realizowany w ramach "Inżynierskiego projektu zespołowego – porównanie metod AI – INF", ma na celu praktyczne zastosowanie i porównanie skuteczności dwóch różnych podejść sztucznej inteligencji w prognozowaniu cen na rynkach finansowych. Koncentrujemy się na przewidywaniu **ceny zamknięcia (`Close`) na następny dzień sesyjny** dla następujących aktywów:

* **AAPL:** Akcje firmy Apple Inc.
* **BTC:** Kryptowaluta Bitcoin (BTC-USD)
* **SP500:** Indeks giełdowy S&P 500 (SPY)

Porównujemy klasyczny model statystyczny **ARIMA** (Autoregressive Integrated Moving Average) z zaawansowaną siecią neuronową **LSTM** (Long Short-Term Memory), wykorzystując również prosty model **Naive Baseline** jako punkt odniesienia. Projekt wykorzystuje popularne biblioteki Pythona do analizy danych, wizualizacji i modelowania.

## Dane 💾

* Projekt operuje na historycznych, dziennych danych **OHLCV** (Open, High, Low, Close, Volume).
* Dane dla wybranych aktywów (AAPL, BTC, SP500) powinny być dostarczone w formacie **CSV**.
* Skrypt jest przystosowany do specyficznego formatu CSV z **trzema wierszami nagłówkowymi** (pierwszy z nazwami kolumn, drugi i trzeci do pominięcia) oraz **datą w pierwszej kolumnie**.

## Główne Funkcjonalności ✨

1.  **Interaktywny Wybór Danych:** Skrypt po uruchomieniu prosi użytkownika o wskazanie 3 plików CSV do analizy za pomocą okna dialogowego.
2.  **Wczytywanie i Czyszczenie Danych:** Automatyczne wczytywanie plików CSV (z obsługą specyficznego formatu), parsowanie dat, sortowanie chronologiczne i usuwanie wierszy z brakującymi wartościami (NaN).
3.  **Inżynieria Cech:** Obliczanie dodatkowych wskaźników technicznych (SMA, RSI, MACD, Daily Return) przy użyciu biblioteki `pandas-ta`, które mogą wspomóc proces prognozowania (szczególnie dla LSTM).
4.  **Podział Chronologiczny:** Dzielenie danych na zbiór **treningowy** (do końca 2023) i **testowy** (od początku 2024) w celu realistycznej oceny modeli.
5.  **Modelowanie i Porównanie:**
    * Implementacja i ewaluacja modelu **Naive Baseline** (prognoza = cena wczoraj).
    * Implementacja, automatyczne strojenie (`auto_arima`) i ewaluacja modelu **ARIMA** (bazującego tylko na historii cen `Close`).
    * Implementacja, trening (na sekwencjach danych z uwzględnieniem wszystkich cech) i ewaluacja modelu **LSTM**. Uwzględnia skalowanie danych wejściowych i docelowych oraz odwracanie skalowania prognoz.
6.  **Ewaluacja:** Porównanie prognoz modeli z rzeczywistymi wartościami na zbiorze testowym przy użyciu metryk regresji: **MAE** (Mean Absolute Error) i **RMSE** (Root Mean Squared Error).
7.  **Wizualizacja Wyników:** Generowanie tabeli podsumowującej wyniki oraz **osobnych wykresów słupkowych** dla każdego aktywa, porównujących błędy MAE i RMSE dla wszystkich trzech modeli.

## Użycie 🚀

1.  Upewnij się, że masz zainstalowane wszystkie wymagane biblioteki (patrz sekcja poniżej).
2.  Przygotuj 3 pliki CSV z danymi historycznymi dla AAPL, BTC i SP500 w oczekiwanym formacie (3 linie nagłówka, data w pierwszej kolumnie).
3.  Uruchom główny skrypt Pythona (np. `MAIN.py`) z terminala:
    ```bash
    python MAIN.py
    ```
4.  W oknie dialogowym, które się pojawi, wybierz przygotowane 3 pliki CSV.
5.  Skrypt przetworzy dane, wytrenuje modele, przeprowadzi ewaluację i zapisze wyniki w folderze **RESULTS**.

## Wymagane Biblioteki 📚

Do poprawnego działania skryptu potrzebne są następujące biblioteki Pythona. Możesz je zainstalować za pomocą pip:

```bash
pip install -r requirements.txt
```

```bash
pip install pandas numpy pandas-ta scikit-learn tensorflow statsmodels pmdarima matplotlib seaborn plotly

