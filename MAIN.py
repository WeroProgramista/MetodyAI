import pandas as pd
import numpy as np
import ta
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
import sys
import os

# Ignorowanie ostrzeżeń dla czytelności
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')


# --- Funkcja do wyboru plików CSV ---
def select_csv_files(num_files=3):
    """Otwiera okno dialogowe do wyboru określonej liczby plików CSV."""
    root = tk.Tk()
    root.withdraw()  # Ukrycie głównego okna tkinter
    print(f"Proszę wybrać dokładnie {num_files} pliki CSV w oknie dialogowym...")
    file_paths = filedialog.askopenfilenames(
        title=f"Wybierz {num_files} pliki CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()

    if len(file_paths) == num_files:
        print(f"Wybrano pliki: {file_paths}")
        return list(file_paths)
    elif len(file_paths) == 0:
        print("Nie wybrano żadnych plików. Przerywanie skryptu.")
        sys.exit()
    else:
        print(f"BŁĄD: Wybrano {len(file_paths)} plików, a wymagane jest {num_files}. Przerywanie skryptu.")
        sys.exit()


# --- Konfiguracja ---
TRAIN_END_DATE = '2023-12-31'
TEST_START_DATE = '2024-01-01'

# Parametry LSTM
TIME_STEPS = 30
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


# --- Wybór Plików przez Użytkownika ---
FILE_PATHS = select_csv_files(num_files=3)

# --- Główna pętla przetwarzania ---
for file_path in FILE_PATHS:
    try:
        base_name = file_path.split('/')[-1].split('\\')[-1]
        dataset_name = base_name.split('_')[0]
        if '.' in dataset_name:
            dataset_name = dataset_name.split('.')[0]
    except IndexError:
        dataset_name = f"Dataset_{FILE_PATHS.index(file_path) + 1}"

    print(f"\n{'=' * 30} Przetwarzanie: {dataset_name} ({file_path}) {'=' * 30}")
    start_time_dataset = time.time()

    # --- 1. Wczytanie Danych ---
    print("1. Wczytywanie danych...")
    try:
        df = pd.read_csv(
            file_path,
            header=0,
            skiprows=[1, 2],
            index_col=0,
            parse_dates=True
        )
        df.index.name = 'Date'

        print(f"Wczytano dane z {file_path}. Rozmiar: {df.shape}")

    except FileNotFoundError:
        print(f"BŁĄD: Plik {file_path} nie został znaleziony.")
        continue
    except Exception as e:
        print(f"BŁĄD podczas wczytywania lub przetwarzania pliku CSV {file_path}: {e}")
        continue

    df.sort_index(inplace=True)

    # --- 2. Inżynieria Cech ---
    print("2. Obliczanie wskaźników technicznych...")
    price_col = 'Close'
    df_features = df.copy()

    df_features['SMA_14'] = ta.trend.sma_indicator(df_features['Close'], window=14)
    df_features['SMA_50'] = ta.trend.sma_indicator(df_features['Close'], window=50)
    df_features['RSI_14'] = ta.momentum.rsi(df_features['Close'], window=14)
    df_features['MACD_12_26_9'] = ta.trend.macd(df_features['Close'])
    df_features['MACDh_12_26_9'] = ta.trend.macd_diff(df_features['Close'])
    df_features['MACDs_12_26_9'] = ta.trend.macd_signal(df_features['Close'])
    df_features['Daily_Return'] = df_features['Close'].pct_change()

    indicator_cols = ['SMA_14', 'SMA_50', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'Daily_Return']

    # --- 3. Tworzenie Zmiennej Docelowej (Regresja) ---
    print("3. Tworzenie zmiennej docelowej (cena zamknięcia następnego dnia)...")
    df_features['Target_Close'] = df_features[price_col].shift(-1)

    # --- 4. Czyszczenie Danych ---
    print("4. Usuwanie wierszy z wartościami NaN...")
    initial_rows = df_features.shape[0]
    columns_to_check_for_nan = indicator_cols + ['Target_Close']
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
        print(
            f"BŁĄD: Jeden ze zbiorów (treningowy: {df_train.shape[0]}, testowy: {df_test.shape[0]}) jest pusty po podziale.")
        continue

    print(f"Rozmiar zbioru treningowego: {df_train.shape}")
    print(f"Rozmiar zbioru testowego: {df_test.shape}")

    X_train = df_train[indicator_cols]
    y_train = df_train['Target_Close']
    X_test = df_test[indicator_cols]
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

    df_test['Close'] = pd.to_numeric(df_test['Close'],
                                     errors='coerce')

    y_pred_naive = df_test[price_col].shift(1)

    y_pred_naive.fillna(y_pred_naive.first_valid_index(), inplace=True)

    print(f"\nTypy danych w y_test przed konwersją: {y_test.dtypes}")
    print(f"Typy danych w y_pred_naive przed konwersją: {y_pred_naive.dtypes}")

    if pd.api.types.is_datetime64_any_dtype(y_test):
        print("y_test zawiera daty. Konwertuję na liczby (dni)...")
        y_test = (y_test - y_test.min()).dt.days

    if pd.api.types.is_datetime64_any_dtype(y_pred_naive):
        print("y_pred_naive zawiera daty. Konwertuję na liczby (dni)...")
        y_pred_naive = (y_pred_naive - y_pred_naive.min()).dt.days

    y_pred_naive = pd.to_numeric(y_pred_naive, errors='coerce')

    y_test = y_test.dropna()
    y_pred_naive = y_pred_naive.dropna()

    if y_test.isnull().any() or y_pred_naive.isnull().any():
        print("Znaleziono brakujące wartości po usunięciu NaN. Zastępuję je średnią.")
        y_test.fillna(y_test.mean(), inplace=True)
        y_pred_naive.fillna(y_pred_naive.mean(), inplace=True)

    print(f"\nTypy danych w y_test po konwersji: {y_test.dtypes}")
    print(f"Typy danych w y_pred_naive po konwersji: {y_pred_naive.dtypes}")

    common_index = y_test.index.intersection(y_pred_naive.index)
    if not common_index.empty:
        y_test_eval_naive = y_test.loc[common_index]
        y_pred_naive_eval = y_pred_naive.loc[common_index]
        evaluate_regression_model('Naive Baseline', dataset_name, y_test_eval_naive, y_pred_naive_eval)

    # --- 8. Model ARIMA ---
    print("\n8. Trenowanie i Ewaluacja Modelu ARIMA...")
    start_time_arima = time.time()
    try:
        auto_model = auto_arima(y_train, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore',
                                trace=False)
        arima_model = ARIMA(y_train, order=auto_model.order)
        arima_result = arima_model.fit()
        y_pred_arima = arima_result.forecast(steps=len(y_test))
        y_pred_arima.index = y_test.index
        common_index_arima = y_test.index.intersection(y_pred_arima.index)
        if not common_index_arima.empty:
            y_test_eval_arima = y_test.loc[common_index_arima]
            y_pred_arima_eval = y_pred_arima.loc[common_index_arima]
            evaluate_regression_model('ARIMA', dataset_name, y_test_eval_arima, y_pred_arima_eval)
    except Exception as e:
        print(f"   BŁĄD podczas trenowania/predykcji ARIMA: {e}")

    print(f"   Czas ARIMA: {time.time() - start_time_arima:.2f} s")

    # --- 9. Model LSTM dla Regresji ---
    print("\n9. Trenowanie i Ewaluacja Modelu LSTM dla Regresji...")
    start_time_lstm = time.time()
    if not X_train_scaled.empty and not y_train_scaled.empty:
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, TIME_STEPS)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, TIME_STEPS)

        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(units=50, return_sequences=False))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(units=1))

        lstm_model.compile(optimizer='adam', loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        lstm_model.fit(X_train_seq, y_train_seq, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, validation_split=0.1,
                       callbacks=[early_stopping])

        y_pred_lstm_scaled = lstm_model.predict(X_test_seq)
        y_pred_lstm = target_scaler.inverse_transform(y_pred_lstm_scaled).flatten()

        start_index_lstm_eval = TIME_STEPS
        end_index_lstm_eval = start_index_lstm_eval + len(y_pred_lstm)

        if end_index_lstm_eval <= len(y_test):
            y_test_eval_lstm = y_test.iloc[start_index_lstm_eval:end_index_lstm_eval]
            evaluate_regression_model('LSTM', dataset_name, y_test_eval_lstm, y_pred_lstm)

    print(f"   Czas LSTM: {time.time() - start_time_lstm:.2f} s")
    print(f"Czas przetwarzania dla {dataset_name}: {time.time() - start_time_dataset:.2f} s")

# --- 10. Podsumowanie Wyników ---
print(f"\n{'=' * 30} Podsumowanie Wyników {'=' * 30}")

if not all_results:
    print("Brak wyników do podsumowania.")
else:
    results_df = pd.DataFrame(all_results)
    results_df = results_df.round(4)
    print(results_df)

    print("\nGenerowanie wykresów porównawczych dla każdego datasetu...")
    if not results_df.dropna(subset=['RMSE', 'MAE']).empty:
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            print("OSTRZEŻENIE: Styl 'seaborn-v0_8-darkgrid' niedostępny, używam stylu domyślnego.")
            pass

        for dataset in results_df['Dataset'].unique():
            df_subset = results_df[results_df['Dataset'] == dataset]

            if not df_subset.dropna(subset=['RMSE', 'MAE']).empty:
                plt.figure(figsize=(8, 5))
                sns.barplot(data=df_subset, x='Model', y='RMSE', palette='viridis')
                plt.title(f'Porównanie RMSE modeli dla: {dataset}')
                plt.ylabel('Root Mean Squared Error (RMSE)')
                plt.xlabel('Model')
                plt.xticks(rotation=0)
                plt.tight_layout()
                plt.show()

                plt.figure(figsize=(8, 5))
                sns.barplot(data=df_subset, x='Model', y='MAE', palette='plasma')
                plt.title(f'Porównanie MAE modeli dla: {dataset}')
                plt.ylabel('Mean Absolute Error (MAE)')
                plt.xlabel('Model')
                plt.xticks(rotation=0)
                plt.tight_layout()
                plt.show()
            else:
                print(f"Brak wystarczających danych liczbowych do wygenerowania wykresów dla datasetu: {dataset}")
    else:
        print("Brak wystarczających danych liczbowych do wygenerowania wykresów porównawczych.")

print("\nAnaliza zakończona.")