# 02_preprocessing.py
# Preliminary preprocessing for CIC-IDS2017

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(data):
    """
    Preprocess CIC-IDS2017 for the proposed IDS experiment.

    Processing includes:
    1. Cleaning invalid values
    2. Removing duplicate records
    3. Separating features and labels
    4. Converting labels into binary classes
    5. Converting categorical features into numerical form
    6. Stratified 70:30 train-test split
    7. MinMaxScaler fitted only on training data

    Binary labels:
        0 = Benign
        1 = Malicious
    """

    data = data.copy()

    # Remove spaces from column names
    data.columns = data.columns.str.strip()

    # Replace infinite values
    data = data.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    # Remove rows containing missing values
    data = data.dropna()

    # Remove duplicate records
    data = data.drop_duplicates()

    # Check Label column
    if "Label" not in data.columns:
        raise ValueError(
            "The dataset must contain a 'Label' column."
        )

    # Clean label values
    data["Label"] = (
        data["Label"]
        .astype(str)
        .str.strip()
    )

    # Convert labels to binary classification
    y = data["Label"].apply(
        lambda label:
        0 if label.upper() == "BENIGN" else 1
    )

    # Remove original label from feature set
    X = data.drop(
        columns=["Label"]
    )

    # Convert categorical columns to numerical values
    X = pd.get_dummies(X)

    # Ensure numerical features
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove invalid values created during conversion
    X = X.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    valid_rows = X.dropna().index

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    # ---------------------------------------------------------
    # Stratified 70:30 train-test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # ---------------------------------------------------------
    # MinMaxScaler
    # Fit only on training data to prevent data leakage.
    # ---------------------------------------------------------

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print("Preprocessing completed.")
    print("------------------------")
    print("Train samples :", X_train_scaled.shape[0])
    print("Test samples  :", X_test_scaled.shape[0])
    print("Features      :", X_train_scaled.shape[1])
    print("Split         : 70:30")
    print("Scaler        : MinMaxScaler")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


if __name__ == "__main__":

    file_path = "CIC-IDS2017.csv"

    data = pd.read_csv(file_path)

    preprocess_data(data)
