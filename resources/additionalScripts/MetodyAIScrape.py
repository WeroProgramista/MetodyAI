import yfinance as yf
import os

# Zakres dat
start_date = "2015-01-01"
end_date = "2025-04-10"

# Ścieżka zapisu
output_dir = r"D:\daneAI"

# Upewnij się, że folder istnieje
os.makedirs(output_dir, exist_ok=True)

# Lista tickerów
tickers = {
    "Apple_AAPL": "AAPL",
    "SP500_SPY": "SPY",
    "Bitcoin_BTCUSD": "BTC-USD"
}

# Pobieranie danych i zapisywanie
for name, ticker in tickers.items():
    df = yf.download(ticker, start=start_date, end=end_date)
    output_path = os.path.join(output_dir, f"{name}_2015_2025.csv")
    df.to_csv(output_path)
    print(f"{name} zapisany jako: {output_path}")
