# 06_deepfool_attack.py
# Preliminary DeepFool-style adversarial attack

import numpy as np


def generate_deepfool(
    model,
    X,
    max_iter=10,
    overshoot=0.02
):
    """
    Generate preliminary DeepFool-style
    adversarial examples for Logistic Regression.

    DeepFool is used only during testing.
    """

    X = np.asarray(
        X,
        dtype=float
    )

    X_adversarial = X.copy()

    weights = model.coef_[0]
    bias = model.intercept_[0]

    weight_norm = np.linalg.norm(
        weights
    )

    if weight_norm == 0:
        return X_adversarial

    original_predictions = (
        model.predict(X)
    )

    for _ in range(max_iter):

        current_predictions = (
            model.predict(
                X_adversarial
            )
        )

        unchanged = (
            current_predictions
            == original_predictions
        )

        if not np.any(unchanged):
            break

        # Distance to Logistic Regression
        # decision boundary
        decision_values = (
            np.dot(
                X_adversarial,
                weights
            )
            + bias
        )

        distance = (
            np.abs(
                decision_values
            )
            / weight_norm
        )

        # Direction toward decision boundary
        direction = (
            -np.sign(
                decision_values
            )[:, np.newaxis]
            * weights
            / weight_norm
        )

        perturbation = (
            distance[:, np.newaxis]
            * direction
            * (1 + overshoot)
        )

        X_adversarial[unchanged] = (
            X_adversarial[unchanged]
            + perturbation[unchanged]
        )

    return X_adversarial


if __name__ == "__main__":

    print("DeepFool Attack")
    print("----------------")
    print(
        "DeepFool is used only during testing."
    )
    print(
        "DeepFool is an UNSEEN attack."
    )
