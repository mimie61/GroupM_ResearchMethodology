# 03_logistic_regression_baseline.py
# Preliminary Logistic Regression baseline

from sklearn.linear_model import LogisticRegression


def create_logistic_regression():
    """
    Create the Logistic Regression classifier.
    """

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    return model


def train_baseline_model(
    X_train,
    y_train
):
    """
    Train the baseline Logistic Regression model
    using clean training data.
    """

    model = create_logistic_regression()

    model.fit(
        X_train,
        y_train
    )

    print(
        "Baseline Logistic Regression "
        "model trained."
    )

    return model


if __name__ == "__main__":

    print(
        "Logistic Regression Baseline"
    )
    print(
        "The baseline model uses clean "
        "CIC-IDS2017 training data."
    )
