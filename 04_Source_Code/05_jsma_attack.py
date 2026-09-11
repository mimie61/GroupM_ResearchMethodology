# 05_jsma_attack.py
# Preliminary JSMA adversarial attack implementation

import numpy as np


def generate_jsma(model, X, target_class, theta=0.01):
    """
    Generate preliminary JSMA-style adversarial examples.

    Parameters:
        model        : trained classification model
        X            : input features
        target_class : target class for the attack
        theta        : feature modification amount

    Returns:
        X_adv        : adversarial examples
    """

    X = np.asarray(X, dtype=float)
    X_adv = X.copy()

    classes = model.classes_

    if target_class not in classes:
        raise ValueError("Target class is not available in the model.")

    target_index = np.where(classes == target_class)[0][0]

    # Logistic Regression coefficients are used to estimate
    # feature influence on the target class.
    feature_influence = model.coef_[target_index]

    # Select influential features
    important_features = np.argsort(
        np.abs(feature_influence)
    )[::-1]

    # Modify the most influential features
    for feature_index in important_features[:10]:
        direction = np.sign(feature_influence[feature_index])

        if direction == 0:
            direction = 1

        X_adv[:, feature_index] += theta * direction

    return X_adv


if __name__ == "__main__":
    print("JSMA attack module")
    print("JSMA is reserved for testing as an unseen attack.")
