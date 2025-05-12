# Projekt Analizy Danych Finansowych (AAPL, BTC, SP500) 📊📈

## Opis Projektu

Ten projekt koncentruje się na analizie danych historycznych oraz potencjalnym modelowaniu predykcyjnym dla wybranych aktywów finansowych:
* **AAPL**: Akcje firmy Apple Inc.
* **BTC**: Kryptowaluta Bitcoin (względem USD lub innej waluty bazowej)
* **SP500**: Indeks giełdowy S&P 500

Projekt wykorzystuje popularne biblioteki Pythona do manipulacji danymi, analizy technicznej, wizualizacji oraz budowy modeli uczenia maszynowego i statystycznych.

## Główne Funkcjonalności ✨

* **Pobieranie Danych**: Zawiera skrypt do pobierania aktualnych danych świecowych (OHLCV) z Yahoo Finance.
* **Przechowywanie Danych**: Dane historyczne przechowywane są w plikach CSV w dedykowanym folderze.
* **Analiza Techniczna**: Obliczanie różnorodnych wskaźników technicznych przy użyciu biblioteki `pandas-ta`.
* **Wizualizacja**: Skrypt do generowania czytelnych wykresów cenowych oraz wskaźników technicznych.
* **Modelowanie (w trakcie rozwoju)**: Eksploracja możliwości predykcji przy użyciu `scikit-learn`, `tensorflow`, `statsmodels` i `pmdarima`.

## Wymagane Biblioteki 📚

Do uruchomienia projektu potrzebne są następujące biblioteki Pythona:

```bash
pip install pandas numpy pandas-ta scikit-learn tensorflow matplotlib seaborn statsmodels pmdarima
