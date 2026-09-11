# 07_fgsm_adversarial_training.py
# Preliminary FGSM adversarial training implementation

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_fgsm_adversarial_model(
    model,
    X_train,
    y_train,
    X_fgsm
):
    """
    Train a Logistic Regression model using
    clean data combined with FGSM adversarial examples.

    Parameters:
        model   : Logistic Regression model
        X_train : clean training data
        y_train : training labels
        X_fgsm  : FGSM adversarial examples

    Returns:
        model : FGSM-trained Logistic Regression model
    """

    # Combine clean training data and FGSM adversarial data
    X_combined = np.concatenate(
        [X_train, X_fgsm],
        axis=0
    )

    # Duplicate labels for adversarial examples
    y_combined = np.concatenate(
        [y_train, y_train],
        axis=0
    )

    # Train the defended model
    model.fit(
        X_combined,
        y_combined
    )

    print("FGSM adversarial training completed.")
    print("Clean training samples:",
          len(X_train))
    print("FGSM adversarial samples:",
          len(X_fgsm))
    print("Total training samples:",
          len(X_combined))

    return model


if __name__ == "__main__":

    print("FGSM Adversarial Training Module")
    print("--------------------------------")
    print("Clean training data is combined")
    print("with FGSM adversarial examples.")
    print("JSMA and DeepFool remain unseen")
    print("during adversarial training.")
