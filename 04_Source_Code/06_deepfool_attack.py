# 06_deepfool_attack.py
# Preliminary DeepFool-style adversarial attack implementation

import numpy as np


def generate_deepfool(model, X, max_iter=10, step_size=0.01):
    """
    Generate preliminary DeepFool-style adversarial examples.

    Parameters:
        model      : trained Logistic Regression model
        X          : input features
        max_iter   : maximum number of iterations
        step_size  : perturbation step size

    Returns:
        X_adv      : adversarial examples
    """

    X = np.asarray(X, dtype=float)
    X_adv = X.copy()

    original_predictions = model.predict(X)

    for _ in range(max_iter):

        current_predictions = model.predict(X_adv)

        # Stop when all samples have changed classification
        if np.all(current_predictions != original_predictions):
            break

        # Logistic Regression decision coefficients
        coefficients = model.coef_

        for i in range(len(X_adv)):

            if current_predictions[i] != original_predictions[i]:
                continue

            # Select the feature with the strongest influence
            feature_scores = np.max(
                np.abs(coefficients),
                axis=0
            )

            feature_index = np.argmax(feature_scores)

            # Apply a small perturbation toward the decision boundary
            direction = np.sign(
                coefficients[0, feature_index]
            )

            if direction == 0:
                direction = 1

            X_adv[i, feature_index] += (
                step_size * direction
            )

    return X_adv


if __name__ == "__main__":
    print("DeepFool attack module")
    print("DeepFool is reserved for testing as an unseen attack.")
