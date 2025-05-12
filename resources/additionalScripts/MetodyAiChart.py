import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num
from mplfinance.original_flavor import candlestick_ohlc
import os

# Ścieżka do katalogu z plikami
base_path = r"D:\daneAI"

# Lista plików CSV
files = {
    "Apple (AAPL)": "Apple_AAPL_2015_2025.csv",
    "Bitcoin (BTC)": "Bitcoin_BTCUSD_2015_2025.csv",
    "S&P 500 (SPY)": "SP500_SPY_2015_2025.csv"
}

# Liczba świec do pokazania
num_candles = 100

for title, filename in files.items():
    file_path = os.path.join(base_path, filename)

    # Wczytanie z właściwym nagłówkiem
    df = pd.read_csv(file_path, header=1)

    # Usuń pierwszy wiersz, który zawiera etykiety np. "Date,,,,,"
    df = df[df[df.columns[0]] != 'Date']

    # Zmień nazwę pierwszej kolumny na "Date"
    df = df.rename(columns={df.columns[0]: 'Date'})

    # Konwertuj Date do formatu datetime z podanym formatem
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d", errors='coerce')

    # Usuń wiersze z błędnymi datami (np. jeśli coś nie zostało sparsowane)
    df = df.dropna(subset=['Date'])

    # Przypisz pozostałe kolumny do właściwych nazw
    df = df.rename(columns={
        df.columns[1]: 'Close',
        df.columns[2]: 'High',
        df.columns[3]: 'Low',
        df.columns[4]: 'Open'
    })

    # Upewnij się, że kolumny mają odpowiednie typy danych
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Usuń ewentualne wiersze z brakami
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    # Przygotowanie danych do wykresu
    df['DateNum'] = date2num(df['Date'])
    ohlc = df[['DateNum', 'Open', 'High', 'Low', 'Close']].iloc[:num_candles]

    # Rysowanie wykresu
    fig, ax = plt.subplots(figsize=(12, 6))
    candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis_date()
    plt.xticks(rotation=45)
    plt.title(f"Wykres świecowy – {title}")
    plt.xlabel("Data")
    plt.ylabel("Cena")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
