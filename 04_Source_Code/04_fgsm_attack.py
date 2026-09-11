# 04_fgsm_attack.py
# Preliminary FGSM adversarial attack implementation

import numpy as np


def generate_fgsm(model, X, y, epsilon=0.01):
    """
    Generate FGSM adversarial examples for a Logistic Regression model.

    Parameters:
        model   : trained Logistic Regression model
        X       : input features
        y       : true labels
        epsilon : perturbation magnitude

    Returns:
        X_adv   : adversarial examples
    """

    X = np.asarray(X, dtype=float)
    X_adv = X.copy()

    # Convert labels into class indices
    classes = model.classes_

    # Calculate probabilities
    probabilities = model.predict_proba(X)

    # Create one-hot encoded labels
    y_encoded = np.zeros_like(probabilities)

    for i, label in enumerate(y):
        class_index = np.where(classes == label)[0][0]
        y_encoded[i, class_index] = 1

    # Calculate gradient of cross-entropy loss
    error = probabilities - y_encoded
    gradient = np.dot(error, model.coef_)

    # FGSM perturbation
    perturbation = epsilon * np.sign(gradient)

    # Generate adversarial examples
    X_adv = X + perturbation

    return X_adv


if __name__ == "__main__":
    print("FGSM attack module")
    print("This module generates adversarial examples")
    print("for FGSM-based adversarial training and testing.")
