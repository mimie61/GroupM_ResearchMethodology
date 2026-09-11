# 03_logistic_regression_baseline.py
# Preliminary Logistic Regression baseline model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_baseline_model(data):
    """
    Train a baseline Logistic Regression model
    using clean CIC-IDS2017 data.
    """

    # Separate features and labels
    X = data.drop("Label", axis=1)
    y = data["Label"]

    # Convert categorical features into numerical values
    X = pd.get_dummies(X)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Normalize features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create Logistic Regression model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    # Train the baseline model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    print("Baseline Logistic Regression Results")
    print("------------------------------------")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-score :", f1)

    return model, scaler


if __name__ == "__main__":
    # Replace this path with the CIC-IDS2017 CSV file location.
    file_path = "CIC-IDS2017.csv"

    data = pd.read_csv(file_path)

    model, scaler = train_baseline_model(data)
