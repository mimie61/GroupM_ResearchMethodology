# 01_data_loading.py
# Preliminary dataset loading for CIC-IDS2017

import pandas as pd


def load_dataset(file_path):
    """
    Load the CIC-IDS2017 dataset from a CSV file.
    """
    data = pd.read_csv(file_path)

    print("Dataset loaded successfully.")
    print("Dataset shape:", data.shape)
    print("\nFirst 5 rows:")
    print(data.head())

    return data


if __name__ == "__main__":
    # Replace this path with the location of the CIC-IDS2017 CSV file.
    file_path = "CIC-IDS2017.csv"

    dataset = load_dataset(file_path)
