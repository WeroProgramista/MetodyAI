import pytest
import pandas as pd
import numpy as np
from ta import trend, momentum
from unittest.mock import patch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MAIN import select_csv_files, create_sequences, evaluate_regression_model

@pytest.fixture
def mock_select_csv_files():
    with patch("tkinter.filedialog.askopenfilenames") as mock_file_dialog:
        mock_file_dialog.return_value = ["test_data_1.csv", "test_data_2.csv", "test_data_3.csv"]
        yield mock_file_dialog

def test_select_csv_files(mock_select_csv_files):
    files = select_csv_files(num_files=3)
    assert len(files) == 3
    assert "test_data_1.csv" in files
    assert "test_data_2.csv" in files
    assert "test_data_3.csv" in files


def test_sma_indicator():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sma = trend.sma_indicator(data, window=3)
    assert len(sma) == len(data)
    assert sma.isnull().head(2).all()
    assert not sma.isnull().iloc[2:].any()


def test_rsi_indicator():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    rsi = momentum.rsi(data, window=3)
    assert len(rsi) == len(data)
    assert rsi.isnull().head(2).all()
    assert not rsi.isnull().iloc[2:].any()


def test_macd_diff():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    macd_diff = trend.macd_diff(data, window_slow=26, window_fast=12, window_sign=9)
    assert len(macd_diff) == len(data)
    assert macd_diff.isnull().head(26).all()
    assert not macd_diff.isnull().iloc[26:].any()

def test_create_sequences():
    X = pd.DataFrame(np.arange(10).reshape(5, 2), columns=["feature1", "feature2"])
    y = pd.Series([1, 2, 3, 4, 5])
    X_seq, y_seq = create_sequences(X, y, time_steps=2)

    assert X_seq.shape[0] == 3
    assert X_seq.shape[1] == 2
    assert y_seq.shape[0] == 3
    assert len(y_seq) == 3


def test_evaluate_regression_model():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.2, 4.0, 4.9])
    result = evaluate_regression_model("Test Model", "Test Dataset", y_true, y_pred)

    assert isinstance(result, dict)
    assert "MAE" in result
    assert "MSE" in result
    assert "RMSE" in result

    assert result["MAE"] == pytest.approx(0.1, rel=1e-2)

    assert result["MSE"] == pytest.approx(0.0140, rel=1e-3)

    assert result["RMSE"] == pytest.approx(0.1183, rel=1e-3)


if __name__ == "__main__":
    pytest.main()
