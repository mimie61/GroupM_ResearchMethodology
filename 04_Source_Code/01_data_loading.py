# 01_data_loading.py
# Preliminary dataset loading for CIC-IDS2017

import pandas as pd


def load_dataset(file_path):
    """
    Load the CIC-IDS2017 dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CIC-IDS2017 CSV file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """

    data = pd.read_csv(file_path)

    print("Dataset loaded successfully.")
    print("Dataset shape:", data.shape)

    return data


def inspect_dataset(data):
    """
    Display basic information about the dataset.
    """

    print("\nDataset Information")
    print("-------------------")
    print("Rows:", data.shape[0])
    print("Columns:", data.shape[1])

    print("\nLabel distribution:")
    if "Label" in data.columns:
        print(data["Label"].value_counts())


if __name__ == "__main__":

    # Replace with the actual CIC-IDS2017 CSV file.
    file_path = "CIC-IDS2017.csv"

    data = load_dataset(file_path)

    inspect_dataset(data)
