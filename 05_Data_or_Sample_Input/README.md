# 05 Data or Sample Input

## Dataset Overview
- **Name:** CIC-IDS2017 Benchmark Dataset
- **Source:** Canadian Institute for Cybersecurity (University of New Brunswick)
- **Format:** Preprocessed CSV / Flow-based tabular data

## Preprocessing Pipeline
1. **Data Cleaning:** Removal of duplicate rows, NaN values, and infinite (`Inf`) values.
2. **Label Encoding:** Binary transformation (0 = Benign Traffic, 1 = Intrusion Attack).
3. **Feature Dropping:** Non-informative metadata removed (IP addresses, ports, timestamps, Flow ID).
4. **Feature Scaling:** Continuous numeric flow features scaled to [0, 1] using `MinMaxScaler`.
5. **Data Splitting:** Stratified 70% Training / 30% Holdout Testing split.
