# 07_fgsm_adversarial_training.py
# Preliminary FGSM adversarial training

import numpy as np

from sklearn.linear_model import LogisticRegression


def train_fgsm_adversarial_model(
    X_train,
    y_train,
    X_fgsm_train
):
    """
    Train Logistic Regression using clean data
    combined with FGSM adversarial examples.

    FGSM is the only attack used during
    adversarial training.
    """

    # Combine clean and FGSM adversarial samples
    X_combined = np.concatenate(
        [
            X_train,
            X_fgsm_train
        ],
        axis=0
    )

    # Adversarial samples retain their
    # original labels
    y_combined = np.concatenate(
        [
            y_train,
            y_train
        ],
        axis=0
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_combined,
        y_combined
    )

    print(
        "FGSM adversarial training completed."
    )

    print(
        "Clean training samples:",
        len(X_train)
    )

    print(
        "FGSM training samples:",
        len(X_fgsm_train)
    )

    return model


if __name__ == "__main__":

    print(
        "FGSM Adversarial Training"
    )
    print(
        "Clean data + FGSM adversarial data"
    )
    print(
        "FGSM = SEEN attack"
    )
    print(
        "JSMA = UNSEEN attack"
    )
    print(
        "DeepFool = UNSEEN attack"
    )
