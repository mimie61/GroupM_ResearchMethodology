# 04_fgsm_attack.py
# Preliminary FGSM adversarial attack

import numpy as np


def generate_fgsm(
    model,
    X,
    y,
    epsilon=0.01
):
    """
    Generate FGSM adversarial examples
    for a binary Logistic Regression model.

    Parameters:
        model    : Trained Logistic Regression model.
        X        : Input feature matrix.
        y        : True binary labels.
        epsilon  : Perturbation magnitude.

    Returns:
        numpy.ndarray: FGSM adversarial examples.
    """

    X = np.asarray(
        X,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    # Probability of malicious class
    probabilities = model.predict_proba(
        X
    )[:, 1]

    # Difference between prediction and true label
    error = probabilities - y

    # Gradient of the binary loss with respect
    # to the input features
    gradient = (
        error[:, np.newaxis]
        * model.coef_[0]
    )

    # FGSM perturbation
    perturbation = (
        epsilon
        * np.sign(gradient)
    )

    # Generate adversarial examples
    X_adversarial = (
        X + perturbation
    )

    return X_adversarial


if __name__ == "__main__":

    print("FGSM Attack")
    print("-----------")
    print(
        "FGSM is used for adversarial "
        "training and testing."
    )
    print(
        "FGSM is the SEEN attack."
    )
