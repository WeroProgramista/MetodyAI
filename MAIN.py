import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import tkinter as tk
from tkinter import filedialog
import sys # To exit if files are not selected correctly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os

# Ignorowanie ostrzeżeń dla czytelności
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR') # Suppress TensorFlow warnings

# --- Funkcja do wyboru plików CSV ---
def select_csv_files(num_files=3):
    """Otwiera okno dialogowe do wyboru określonej liczby plików CSV."""
    root = tk.Tk()
    root.withdraw() # Ukrycie głównego okna tkinter
    print(f"Proszę wybrać dokładnie {num_files} pliki CSV w oknie dialogowym...")
    file_paths = filedialog.askopenfilenames(
        title=f"Wybierz {num_files} pliki CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy() # Zamknięcie okna tkinter po wyborze

    if len(file_paths) == num_files:
        print(f"Wybrano pliki: {file_paths}")
        return list(file_paths)
    elif len(file_paths) == 0:
        print("Nie wybrano żadnych plików. Przerywanie skryptu.")
        sys.exit() # Zakończ skrypt, jeśli użytkownik anulował
    else:
        print(f"BŁĄD: Wybrano {len(file_paths)} plików, a wymagane jest {num_files}. Przerywanie skryptu.")
        sys.exit() # Zakończ skrypt, jeśli wybrano złą liczbę plików

# --- Konfiguracja ---
# Zakres dat dla podziału trening/test
TRAIN_END_DATE = '2023-12-31'
TEST_START_DATE = '2024-01-01'

# Parametry LSTM
TIME_STEPS = 30 # Długość sekwencji wejściowej
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# --- Słownik na wyniki ---
all_results = []

# --- Funkcja do Ewaluacji Modeli Regresyjnych ---
def evaluate_regression_model(name, dataset_name, y_true, y_pred):


    """Oblicza i drukuje metryki regresji, zapisuje wyniki."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n--- Wyniki dla: {name} ({dataset_name}) ---")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Zapisanie wyników
    all_results.append({
        'Dataset': dataset_name,
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    })
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

# --- Funkcja do tworzenia sekwencji dla LSTM ---
def create_sequences(X, y, time_steps=1):
    """Tworzy sekwencje danych dla modeli sekwencyjnych (LSTM)."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def plot_combined_predictions(df_plot, y_true, y_naive, y_arima, y_lstm, title):
    fig = go.Figure()

    # Wykres świecowy (OHLC)
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'],
        high=df_plot['High'],
        low=df_plot['Low'],
        close=df_plot['Close'],
        name='Dane historyczne',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # Rzeczywiste wartości
    fig.add_trace(go.Scatter(
        x=y_true.index,
        y=y_true.values,
        mode='lines',
        name='Rzeczywiste',
        line=dict(color='blue')
    ))

    # Naive
    fig.add_trace(go.Scatter(
        x=y_naive.index,
        y=y_naive.values,
        mode='lines',
        name='Naive',
        line=dict(color='gray', dash='dot')
    ))

    # ARIMA
    fig.add_trace(go.Scatter(
        x=y_arima.index,
        y=y_arima.values,
        mode='lines',
        name='ARIMA',
        line=dict(color='purple', dash='dash')
    ))

    # LSTM
    fig.add_trace(go.Scatter(
        x=y_lstm.index,
        y=y_lstm.values,
        mode='lines',
        name='LSTM',
        line=dict(color='orange')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Data',
        yaxis_title='Cena',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=600
    )

    return fig

# Utworzenie katalogu na wyniki
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Wybór Plików przez Użytkownika ---
FILE_PATHS = select_csv_files(num_files=3)

# --- Główna Pętla Przetwarzania ---
for file_path in FILE_PATHS:
    # Pobranie nazwy datasetu z nazwy pliku (prosta metoda)
    try:
        # Próba wydobycia nazwy przed pierwszym '_' lub '.'
        base_name = file_path.split('/')[-1].split('\\')[-1] # Get filename
        dataset_name = base_name.split('_')[0]
        if '.' in dataset_name: # Handle cases like 'SP500.csv' if no underscore
             dataset_name = dataset_name.split('.')[0]
    except IndexError:
        dataset_name = f"Dataset_{FILE_PATHS.index(file_path)+1}" # Fallback name

    print(f"\n{'='*30} Przetwarzanie: {dataset_name} ({file_path}) {'='*30}")
    start_time_dataset = time.time()

    # --- 1. Wczytanie Danych (ZMODYFIKOWANE) ---
    print("1. Wczytywanie danych...")
    try:
        # Wczytaj CSV:
        # header=0: Użyj pierwszego wiersza (np. Price,Close,...) jako nagłówków.
        # skiprows=[1, 2]: Pomiń drugi (np. Ticker) i trzeci (np. Date,,,) wiersz - DOSTOSUJ jeśli trzeba.
        # index_col=0: Użyj pierwszej kolumny (tej pod nagłówkiem 'Price', która zawiera daty) jako indeksu.
        # parse_dates=True: Spróbuj przekonwertować wartości w kolumnie indeksu na daty.
        df = pd.read_csv(
            file_path,
            header=0,       # Pierwszy wiersz to nagłówki kolumn
            skiprows=[1, 2],  # Pomiń wiersze 1 i 2 (licząc od 0)
            index_col=0,    # Pierwsza kolumna (indeks 0) zawiera daty
            parse_dates=True # Próbuj sparsować indeks jako daty
        )
        # Zmień nazwę indeksu na 'Date' dla spójności z resztą kodu
        df.index.name = 'Date'

        # Sprawdź, czy wczytano oczekiwane kolumny (Close, High, Low, Open, Volume)
        required_cols_mapping = {'Close': 'Close', 'High': 'High', 'Low': 'Low', 'Open': 'Open', 'Volume': 'Volume'}
        missing_cols = [col for col in required_cols_mapping.values() if col not in df.columns]
        if missing_cols:
             print(f"OSTRZEŻENIE: Brak kolumn {missing_cols} w pliku {file_path}. Dostępne kolumny: {df.columns.tolist()}")

        print(f"Wczytano dane z {file_path}. Rozmiar: {df.shape}")
        # print("Pierwsze 5 wierszy wczytanych danych:") # Opcjonalnie można wyłączyć
        # print(df.head())

    except FileNotFoundError:
        print(f"BŁĄD: Plik {file_path} nie został znaleziony.")
        continue
    except Exception as e:
        print(f"BŁĄD podczas wczytywania lub przetwarzania pliku CSV {file_path}: {e}")
        continue

    df.sort_index(inplace=True)
    # Proste uzupełnienie NaN w Volume
    volume_col = 'Volume'
    if volume_col not in df.columns:
        print(f"OSTRZEŻENIE: Brak kolumny '{volume_col}'. Zostanie zignorowana.")
    elif df[volume_col].isnull().any():
        df[volume_col].fillna(method='ffill', inplace=True)
        df[volume_col].fillna(0, inplace=True)

    # --- 2. Inżynieria Cech ---
    print("2. Obliczanie wskaźników technicznych...")
    price_col = 'Close'
    if price_col not in df.columns:
        print(f"BŁĄD: Brak wymaganej kolumny '{price_col}' w pliku {file_path}.")
        continue

    df_features = df.copy()
    base_feature_columns = ['Open', 'High', 'Low', 'Close']
    if volume_col in df.columns:
        base_feature_columns.append(volume_col)

    missing_base_cols = [col for col in ['Open', 'High', 'Low', 'Close'] if col not in df.columns]
    if missing_base_cols:
        print(f"BŁĄD: Brak wymaganych kolumn {missing_base_cols} w pliku {file_path}.")
        continue

    feature_columns = list(base_feature_columns)
    try:
      df_features.ta.sma(close=price_col, length=14, append=True, col_names=('SMA_14',))
      df_features.ta.sma(close=price_col, length=50, append=True, col_names=('SMA_50',))
      df_features.ta.rsi(close=price_col, length=14, append=True, col_names=('RSI_14',))
      df_features.ta.macd(close=price_col, fast=12, slow=26, signal=9, append=True, col_names=('MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'))
      df_features['Daily_Return'] = df_features[price_col].pct_change()
      indicator_cols = ['SMA_14', 'SMA_50', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'Daily_Return']
      feature_columns.extend(indicator_cols)
    except Exception as e:
        print(f"OSTRZEŻENIE: Błąd podczas obliczania wskaźników technicznych: {e}. Kontynuacja bez nich.")
        feature_columns = list(base_feature_columns)


    # --- 3. Tworzenie Zmiennej Docelowej (Regresja) ---
    print("3. Tworzenie zmiennej docelowej (cena zamknięcia następnego dnia)...")
    df_features['Target_Close'] = df_features[price_col].shift(-1)

    # --- 4. Czyszczenie Danych ---
    print("4. Usuwanie wierszy z wartościami NaN...")
    initial_rows = df_features.shape[0]
    existing_feature_columns = [col for col in feature_columns if col in df_features.columns]
    columns_to_check_for_nan = existing_feature_columns + ['Target_Close']
    df_cleaned = df_features.dropna(subset=columns_to_check_for_nan).copy()
    final_rows = df_cleaned.shape[0]
    print(f"Usunięto {initial_rows - final_rows} wierszy z NaN.")

    if df_cleaned.empty or final_rows < TIME_STEPS + 50:
        print("BŁĄD: Niewystarczająca ilość danych po czyszczeniu.")
        continue

    # --- 5. Podział Danych na Treningowe i Testowe ---
    print("5. Dzielenie danych na zbiór treningowy i testowy...")
    df_train = df_cleaned[:TRAIN_END_DATE]
    df_test = df_cleaned[TEST_START_DATE:]

    if df_train.empty or df_test.empty:
        if df_train.empty and not df_cleaned.empty:
             split_index = int(len(df_cleaned) * 0.8)
             df_train = df_cleaned.iloc[:split_index]
             df_test = df_cleaned.iloc[split_index:]
             print(f"OSTRZEŻENIE: Domyślna data podziału nie zadziałała. Użyto podziału 80/20.")
        elif df_test.empty:
             print(f"OSTRZEŻENIE: Brak danych testowych po {TEST_START_DATE}. Sprawdź zakres dat.")
             continue
        else:
             print(f"BŁĄD: Jeden ze zbiorów (treningowy: {df_train.shape[0]}, testowy: {df_test.shape[0]}) jest pusty po podziale.")
             continue

    print(f"Rozmiar zbioru treningowego: {df_train.shape}")
    print(f"Rozmiar zbioru testowego: {df_test.shape}")

    X_train = df_train[[col for col in existing_feature_columns if col in df_train.columns]]
    y_train = df_train['Target_Close']
    X_test = df_test[[col for col in existing_feature_columns if col in df_test.columns]]
    y_test = df_test['Target_Close']

    # --- 6. Skalowanie Cech i Targetu (dla LSTM) ---
    print("6. Skalowanie cech i targetu...")
    feature_scaler = MinMaxScaler()
    X_train_scaled_np = feature_scaler.fit_transform(X_train.values)
    X_test_scaled_np = feature_scaler.transform(X_test.values)
    X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_np, columns=X_test.columns, index=X_test.index)

    target_scaler = MinMaxScaler()
    y_train_scaled_np = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled_np = target_scaler.transform(y_test.values.reshape(-1, 1))
    y_train_scaled = pd.Series(y_train_scaled_np.flatten(), index=y_train.index)
    y_test_scaled = pd.Series(y_test_scaled_np.flatten(), index=y_test.index)


    # --- 7. Model Bazowy (Naive Baseline) ---
    print("\n7. Ewaluacja Modelu Bazowego (Naive)...")
    y_pred_naive = df_test[price_col].shift(1)
    if y_pred_naive.isnull().any():
        first_valid_index = y_pred_naive.first_valid_index()
        if first_valid_index is not None:
             y_pred_naive.fillna(y_pred_naive[first_valid_index], inplace=True)
        else:
             y_pred_naive.fillna(0, inplace=True)

    common_index = y_test.index.intersection(y_pred_naive.index)
    if common_index.empty:
         print("Brak wspólnych danych (indeksów) do ewaluacji modelu Naive.")
         all_results.append({'Dataset': dataset_name, 'Model': 'Naive Baseline', 'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan})
    else:
        y_test_eval_naive = y_test.loc[common_index]
        y_pred_naive_eval = y_pred_naive.loc[common_index]
        evaluate_regression_model('Naive Baseline', dataset_name, y_test_eval_naive, y_pred_naive_eval)
      

   



    # --- 8. Model ARIMA ---
    print("\n8. Trenowanie i Ewaluacja Modelu ARIMA...")
    start_time_arima = time.time()
    try:
        print("   Uruchamianie auto_arima (może chwilę potrwać)...")
        if y_train.empty:
            raise ValueError("Zbiór treningowy y_train jest pusty dla ARIMA.")

        auto_model = auto_arima(y_train,
                                seasonal=False,
                                stepwise=True,
                                suppress_warnings=True,
                                error_action='ignore',
                                trace=False,
                                max_p=5, max_q=5, max_d=2)

        print(f"   Wybrane parametry ARIMA: {auto_model.order}")

        arima_model = ARIMA(y_train, order=auto_model.order)
        arima_result = arima_model.fit()
        print("   Model ARIMA wytrenowany.")

        if y_test.empty:
            raise ValueError("Zbiór testowy y_test jest pusty dla prognozy ARIMA.")

        y_pred_arima = arima_result.forecast(steps=len(y_test))
        y_pred_arima.index = y_test.index

        common_index_arima = y_test.index.intersection(y_pred_arima.index)
        if common_index_arima.empty:
            print("Brak wspólnych danych (indeksów) do ewaluacji modelu ARIMA.")
            all_results.append({'Dataset': dataset_name, 'Model': 'ARIMA', 'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan})
        else:
            y_test_eval_arima = y_test.loc[common_index_arima]
            y_pred_arima_eval = y_pred_arima.loc[common_index_arima]
            evaluate_regression_model('ARIMA', dataset_name, y_test_eval_arima, y_pred_arima_eval)

       


    except Exception as e:
        print(f"   BŁĄD podczas trenowania/predykcji ARIMA: {e}")
        all_results.append({'Dataset': dataset_name, 'Model': 'ARIMA', 'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan})

    print(f"   Czas ARIMA: {time.time() - start_time_arima:.2f} s")


    # --- 9. Model LSTM dla Regresji ---
    print("\n9. Trenowanie i Ewaluacja Modelu LSTM dla Regresji...")
    start_time_lstm = time.time()
    if X_train_scaled.empty or y_train_scaled.empty:
         print(f"   BŁĄD: Puste dane treningowe (X lub y) dla sekwencji LSTM.")
         all_results.append({'Dataset': dataset_name, 'Model': 'LSTM', 'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan})
    else:
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, TIME_STEPS)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, TIME_STEPS)

        print(f"   Kształt sekwencji treningowych X: {X_train_seq.shape}, y: {y_train_seq.shape}")
        print(f"   Kształt sekwencji testowych X: {X_test_seq.shape}, y: {y_test_seq.shape}")

        if X_train_seq.shape[0] < 10 or X_test_seq.shape[0] == 0:
            print(f"   BŁĄD: Niewystarczająca ilość sekwencji dla LSTM.")
            all_results.append({'Dataset': dataset_name, 'Model': 'LSTM', 'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan})
        else:
            lstm_model = Sequential()
            lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
            lstm_model.add(Dropout(0.2))
            lstm_model.add(LSTM(units=50, return_sequences=False))
            lstm_model.add(Dropout(0.2))
            lstm_model.add(Dense(units=1))

            lstm_model.compile(optimizer='adam', loss='mean_squared_error')

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            print("   Rozpoczynanie treningu LSTM...")
            history = lstm_model.fit(
                X_train_seq, y_train_seq,
                epochs=LSTM_EPOCHS,
                batch_size=LSTM_BATCH_SIZE,
                validation_split=0.1,
                callbacks=[early_stopping],
                shuffle=False,
                verbose=0
            )
            print("   Zakończono trening LSTM.")

            y_pred_lstm_scaled = lstm_model.predict(X_test_seq, verbose=0)
            y_pred_lstm = target_scaler.inverse_transform(y_pred_lstm_scaled).flatten()

            start_index_lstm_eval = TIME_STEPS
            end_index_lstm_eval = start_index_lstm_eval + len(y_pred_lstm)

            if not y_test.empty and end_index_lstm_eval <= len(y_test):
                y_test_eval_lstm = y_test.iloc[start_index_lstm_eval:end_index_lstm_eval]
                if len(y_test_eval_lstm) == len(y_pred_lstm):
                    evaluate_regression_model('LSTM', dataset_name, y_test_eval_lstm, y_pred_lstm)
                  

                else:
                      print(f"   OSTRZEŻENIE: Niezgodność długości y_test_eval_lstm ({len(y_test_eval_lstm)}) i y_pred_lstm ({len(y_pred_lstm)}). Pomijanie ewaluacji LSTM.")
                      all_results.append({'Dataset': dataset_name, 'Model': 'LSTM', 'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan})
            else:
                 print(f"   OSTRZEŻENIE: Indeks poza zakresem lub pusty y_test dla ewaluacji LSTM. Długość y_test: {len(y_test)}, wymagany zakres: {start_index_lstm_eval}-{end_index_lstm_eval}. Pomijanie ewaluacji LSTM.")
                 all_results.append({'Dataset': dataset_name, 'Model': 'LSTM', 'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan})


    print(f"   Czas LSTM: {time.time() - start_time_lstm:.2f} s")
    print(f"Czas przetwarzania dla {dataset_name}: {time.time() - start_time_dataset:.2f} s")

    try:
        # Przygotowanie danych i indeksów wspólnych do wykresu świecowego
        common_index = y_test.index.intersection(y_pred_naive.index).intersection(y_pred_arima.index)
        common_index = common_index.intersection(y_test.index[TIME_STEPS:])  

        if not common_index.empty:
            df_plot = df_test.loc[common_index]
            y_true_final = y_test.loc[common_index]
            y_naive_final = y_pred_naive.loc[common_index]
            y_arima_final = y_pred_arima.loc[common_index]
            y_lstm_final = pd.Series(y_pred_lstm, index=y_test.iloc[TIME_STEPS:TIME_STEPS+len(y_pred_lstm)].index)
            y_lstm_final = y_lstm_final.loc[common_index]

            fig = plot_combined_predictions(df_plot, y_true_final, y_naive_final, y_arima_final, y_lstm_final,
                                            title=f"Candlestick + Predykcje: {dataset_name}")
            fig.write_html(os.path.join(RESULTS_DIR, f"{dataset_name}_combined_forecast.html"))

    except Exception as e:
        print(f"   OSTRZEŻENIE: Nie udało się wygenerować wykresu zbiorczego dla {dataset_name}: {e}")


# --- 10. Podsumowanie Wyników ---
print(f"\n{'='*30} Podsumowanie Wyników {'='*30}")

# --- Utwórz folder 'results', jeśli nie istnieje ---



if not all_results:
    print("Brak wyników do podsumowania.")
else:
    results_df = pd.DataFrame(all_results)
    results_df = results_df.round(4) # Zaokrąglenie metryk
    print(results_df)
    results_df.to_csv(os.path.join(RESULTS_DIR, "podsumowanie_wynikow.csv"), index=False)

    # --- Wizualizacja Porównawcza (ZMODYFIKOWANA) ---
    print("\nGenerowanie wykresów porównawczych dla każdego datasetu...")
    # Sprawdzenie czy są dane do wygenerowania wykresów
    if not results_df.dropna(subset=['RMSE', 'MAE']).empty:
        try:
            plt.style.use('seaborn-v0_8-darkgrid') # Użyj dostępnego stylu
        except OSError:
            print("OSTRZEŻENIE: Styl 'seaborn-v0_8-darkgrid' niedostępny, używam stylu domyślnego.")
            pass # Użyj domyślnego stylu, jeśli ten nie jest dostępny

        # Iteruj po unikalnych nazwach datasetów w wynikach
        for dataset in results_df['Dataset'].unique():
            # Filtruj wyniki dla bieżącego datasetu
            df_subset = results_df[results_df['Dataset'] == dataset]

            if not df_subset.dropna(subset=['RMSE', 'MAE']).empty:
                # Wykres RMSE dla bieżącego datasetu
                plt.figure(figsize=(8, 5)) # Mniejszy rozmiar dla pojedynczego wykresu
                sns.barplot(data=df_subset, x='Model', y='RMSE', palette='viridis') # Użyj 'Model' na osi X
                plt.title(f'Porównanie RMSE modeli dla: {dataset}')
                plt.ylabel('Root Mean Squared Error (RMSE)')
                plt.xlabel('Model')
                plt.xticks(rotation=0)
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTS_DIR, f"{dataset}_RMSE.png"))
                plt.close()

                # Wykres MAE dla bieżącego datasetu
                plt.figure(figsize=(8, 5))
                sns.barplot(data=df_subset, x='Model', y='MAE', palette='plasma')
                plt.title(f'Porównanie MAE modeli dla: {dataset}')
                plt.ylabel('Mean Absolute Error (MAE)')
                plt.xlabel('Model')
                plt.xticks(rotation=0)
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTS_DIR, f"{dataset}_MAE.png"))
                plt.close()
            else:
                print(f"Brak wystarczających danych liczbowych do wygenerowania wykresów dla datasetu: {dataset}")

    else:
        print("Brak wystarczających danych liczbowych do wygenerowania wykresów porównawczych.")



print(f"\nWszystkie wykresy zapisano w katalogu '{RESULTS_DIR}'.")
print(f"Podsumowanie wyników zapisano w 'podsumowanie_wynikow.csv' w katalogu '{RESULTS_DIR}'.")
print("\nAnaliza zakończona.")
