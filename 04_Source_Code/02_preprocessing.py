# 02_preprocessing.py
# Preliminary data preprocessing for CIC-IDS2017

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(data):
    """
    Clean and preprocess the CIC-IDS2017 dataset.
    """

    # Replace infinite values with missing values
    data = data.replace([float("inf"), float("-inf")], pd.NA)

    # Remove rows containing missing values
    data = data.dropna()

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Separate features and labels
    X = data.drop("Label", axis=1)
    y = data["Label"]

    # Convert categorical features into numerical values
    X = pd.get_dummies(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Feature normalization
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Preprocessing completed.")
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    # Replace this path with the location of the CIC-IDS2017 CSV file.
    file_path = "CIC-IDS2017.csv"

    data = pd.read_csv(file_path)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
