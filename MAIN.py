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

# Ignorowanie ostrzeżeń dla czytelności
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR') # Suppress TensorFlow warnings

# --- Konfiguracja ---
# Lista plików do przetworzenia
FILE_PATHS = ['Apple_AAPL_2015_2025.csv',
              'Bitcoin_BTCUSD_2015_2025.csv',
              'SP500_SPY_2015_2025.csv']

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

# --- Główna Pętla Przetwarzania ---
for file_path in FILE_PATHS:
    dataset_name = file_path.split('_')[0] # Pobranie nazwy datasetu z nazwy pliku
    print(f"\n{'='*30} Przetwarzanie: {dataset_name} ({file_path}) {'='*30}")
    start_time_dataset = time.time()

    # --- 1. Wczytanie Danych ---
    print("1. Wczytywanie danych...")
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        print(f"Wczytano dane. Rozmiar: {df.shape}")
    except FileNotFoundError:
        print(f"BŁĄD: Plik {file_path} nie został znaleziony.")
        continue # Przejdź do następnego pliku
    except Exception as e:
        print(f"BŁĄD podczas wczytywania pliku CSV: {e}")
        continue

    df.sort_index(inplace=True)
    # Proste uzupełnienie NaN w Volume
    if df['Volume'].isnull().any():
        df['Volume'].fillna(method='ffill', inplace=True)
        df['Volume'].fillna(0, inplace=True)

    # --- 2. Inżynieria Cech ---
    print("2. Obliczanie wskaźników technicznych...")
    price_col = 'Close'
    df_features = df.copy()
    try:
      df_features.ta.sma(length=14, append=True, col_names=('SMA_14',))
      df_features.ta.sma(length=50, append=True, col_names=('SMA_50',))
      df_features.ta.rsi(length=14, append=True, col_names=('RSI_14',))
      df_features.ta.macd(fast=12, slow=26, signal=9, append=True, col_names=('MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'))
      df_features['Daily_Return'] = df_features[price_col].pct_change()
    except Exception as e:
        print(f"BŁĄD podczas obliczania wskaźników: {e}")
        # Kontynuuj bez wskaźników, jeśli wystąpi błąd
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    else:
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                           'SMA_14', 'SMA_50', 'RSI_14',
                           'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
                           'Daily_Return']

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

    if df_cleaned.empty or final_rows < TIME_STEPS + 50: # Podstawowe sprawdzenie ilości danych
        print("BŁĄD: Niewystarczająca ilość danych po czyszczeniu.")
        continue

    # --- 5. Podział Danych na Treningowe i Testowe ---
    print("5. Dzielenie danych na zbiór treningowy i testowy...")
    df_train = df_cleaned[:TRAIN_END_DATE]
    df_test = df_cleaned[TEST_START_DATE:]

    if df_train.empty or df_test.empty:
        print(f"BŁĄD: Jeden ze zbiorów (treningowy: {df_train.shape[0]}, testowy: {df_test.shape[0]}) jest pusty.")
        continue

    print(f"Rozmiar zbioru treningowego: {df_train.shape}")
    print(f"Rozmiar zbioru testowego: {df_test.shape}")

    # Oddzielenie cech (X) i targetu (y)
    X_train = df_train[existing_feature_columns]
    y_train = df_train['Target_Close']
    X_test = df_test[existing_feature_columns]
    y_test = df_test['Target_Close']

    # --- 6. Skalowanie Cech i Targetu (dla LSTM) ---
    print("6. Skalowanie cech i targetu...")
    # Skaler dla cech wejściowych (X)
    feature_scaler = MinMaxScaler()
    X_train_scaled_np = feature_scaler.fit_transform(X_train.values)
    X_test_scaled_np = feature_scaler.transform(X_test.values)
    X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=existing_feature_columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_np, columns=existing_feature_columns, index=X_test.index)

    # Skaler dla zmiennej docelowej (y) - WAŻNE dla LSTM
    target_scaler = MinMaxScaler()
    # Skalujemy y_train i y_test osobno, ale scaler jest dopasowany tylko na y_train
    # reshape(-1, 1) jest potrzebny, bo scaler oczekuje danych 2D
    y_train_scaled_np = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled_np = target_scaler.transform(y_test.values.reshape(-1, 1))
    # Konwersja z powrotem do Series dla spójności (opcjonalnie)
    y_train_scaled = pd.Series(y_train_scaled_np.flatten(), index=y_train.index)
    y_test_scaled = pd.Series(y_test_scaled_np.flatten(), index=y_test.index)


    # --- 7. Model Bazowy (Naive Baseline) ---
    print("\n7. Ewaluacja Modelu Bazowego (Naive)...")
    # Przewiduj cenę jutro jako cenę dzisiaj
    y_pred_naive = df_test[price_col].shift(1) # Używamy ceny zamknięcia z dnia poprzedniego
    y_pred_naive.fillna(method='bfill', inplace=True) # Wypełnienie pierwszej wartości NaN
    evaluate_regression_model('Naive Baseline', dataset_name, y_test, y_pred_naive)


    # --- 8. Model ARIMA ---
    print("\n8. Trenowanie i Ewaluacja Modelu ARIMA...")
    start_time_arima = time.time()
    try:
        # Używamy auto_arima do znalezienia najlepszych parametrów (p,d,q) dla danych treningowych 'Close'
        # Skupiamy się na prognozie samej ceny, bez dodatkowych wskaźników (model univariate)
        print("   Uruchamianie auto_arima (może chwilę potrwać)...")
        auto_model = auto_arima(y_train, # Używamy nieskalowanego y_train
                                seasonal=False, # Zakładamy brak sezonowości (można zmienić na True jeśli podejrzewamy)
                                stepwise=True,
                                suppress_warnings=True,
                                error_action='ignore',
                                trace=False, # Ustaw na True, aby zobaczyć kroki auto_arima
                                max_p=5, max_q=5, max_d=2) # Ograniczenie przestrzeni poszukiwań

        print(f"   Wybrane parametry ARIMA: {auto_model.order}")

        # Trenowanie finalnego modelu ARIMA na danych treningowych z najlepszymi parametrami
        # Używamy statsmodels.tsa.arima.model.ARIMA
        arima_model = ARIMA(y_train, order=auto_model.order)
        arima_result = arima_model.fit()
        print("   Model ARIMA wytrenowany.")

        # Generowanie prognoz na okres testowy
        # forecast() jest preferowane dla prognoz poza próbą
        y_pred_arima = arima_result.forecast(steps=len(y_test))
        # Upewniamy się, że indeksy pasują
        y_pred_arima.index = y_test.index

        evaluate_regression_model('ARIMA', dataset_name, y_test, y_pred_arima)

    except Exception as e:
        print(f"   BŁĄD podczas trenowania/predykcji ARIMA: {e}")
        # Zapisanie pustych wyników, jeśli ARIMA się nie powiedzie
        all_results.append({'Dataset': dataset_name, 'Model': 'ARIMA', 'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan})

    print(f"   Czas ARIMA: {time.time() - start_time_arima:.2f} s")


    # --- 9. Model LSTM dla Regresji ---
    print("\n9. Trenowanie i Ewaluacja Modelu LSTM dla Regresji...")
    start_time_lstm = time.time()
    # Tworzenie sekwencji dla LSTM
    # Używamy przeskalowanych cech (X) i przeskalowanego targetu (y)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, TIME_STEPS)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, TIME_STEPS)

    print(f"   Kształt sekwencji treningowych X: {X_train_seq.shape}, y: {y_train_seq.shape}")
    print(f"   Kształt sekwencji testowych X: {X_test_seq.shape}, y: {y_test_seq.shape}")

    if X_train_seq.shape[0] < 10 or X_test_seq.shape[0] == 0:
        print(f"   BŁĄD: Niewystarczająca ilość sekwencji dla LSTM.")
        all_results.append({'Dataset': dataset_name, 'Model': 'LSTM', 'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan})
    else:
        # Definicja modelu LSTM dla regresji
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(units=50, return_sequences=False))
        lstm_model.add(Dropout(0.2))
        # Warstwa wyjściowa Dense z 1 neuronem (bez aktywacji lub liniowa dla regresji)
        lstm_model.add(Dense(units=1))

        # Kompilacja modelu dla regresji
        lstm_model.compile(optimizer='adam', loss='mean_squared_error') # MSE jako funkcja straty

        # Callback EarlyStopping monitorujący val_loss
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Trening modelu
        print("   Rozpoczynanie treningu LSTM...")
        history = lstm_model.fit(
            X_train_seq, y_train_seq,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            validation_split=0.1,
            callbacks=[early_stopping],
            shuffle=False,
            verbose=0 # Wyłącz logowanie epok dla czystszego outputu
        )
        print("   Zakończono trening LSTM.")

        # Predykcje na danych testowych (wynik będzie przeskalowany)
        y_pred_lstm_scaled = lstm_model.predict(X_test_seq, verbose=0)

        # Odwrócenie skalowania predykcji do oryginalnej skali cen
        y_pred_lstm = target_scaler.inverse_transform(y_pred_lstm_scaled).flatten()

        # Przygotowanie y_test do porównania (oryginalne wartości, dopasowane do długości predykcji)
        # y_test_seq odpowiada y_pred_lstm
        # Potrzebujemy oryginalnych wartości y_test odpowiadających y_test_seq
        # Indeks y_test_seq zaczyna się od TIME_STEPS, więc bierzemy y_test od tego miejsca
        y_test_eval_lstm = y_test.iloc[TIME_STEPS:TIME_STEPS+len(y_pred_lstm)]

        # Ewaluacja na oryginalnej skali
        evaluate_regression_model('LSTM', dataset_name, y_test_eval_lstm, y_pred_lstm)

        # Opcjonalnie: Wykres historii treningu LSTM
        # plt.figure(figsize=(6, 3))
        # plt.plot(history.history['loss'], label='Training Loss')
        # plt.plot(history.history['val_loss'], label='Validation Loss')
        # plt.title(f'LSTM Training & Validation Loss - {dataset_name}')
        # plt.legend()
        # plt.show()

    print(f"   Czas LSTM: {time.time() - start_time_lstm:.2f} s")
    print(f"Czas przetwarzania dla {dataset_name}: {time.time() - start_time_dataset:.2f} s")


# --- 10. Podsumowanie Wyników ---
print(f"\n{'='*30} Podsumowanie Wyników {'='*30}")

if not all_results:
    print("Brak wyników do podsumowania.")
else:
    results_df = pd.DataFrame(all_results)
    # Formatowanie dla lepszej czytelności
    results_df = results_df.round(4) # Zaokrąglenie metryk
    print(results_df)

    # --- Wizualizacja Porównawcza ---
    print("\nGenerowanie wykresów porównawczych...")
    plt.style.use('seaborn-v0_8-darkgrid') # Ustawienie stylu wykresów

    # Porównanie RMSE dla różnych modeli i datasetów
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='Dataset', y='RMSE', hue='Model', palette='viridis')
    plt.title('Porównanie RMSE modeli dla różnych zbiorów danych')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.xlabel('Zbiór danych')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Porównanie MAE dla różnych modeli i datasetów
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='Dataset', y='MAE', hue='Model', palette='plasma')
    plt.title('Porównanie MAE modeli dla różnych zbiorów danych')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.xlabel('Zbiór danych')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

print("\nAnaliza zakończona.")