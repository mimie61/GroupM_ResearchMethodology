# 05_jsma_attack.py
# Preliminary JSMA-style adversarial attack

import numpy as np


def generate_jsma(
    model,
    X,
    target_class=0,
    theta=0.01,
    max_features=10
):
    """
    Generate preliminary JSMA-style
    adversarial examples.

    Target class:
        0 = Benign

    JSMA is used only during testing
    in the proposed research.
    """

    X = np.asarray(
        X,
        dtype=float
    )

    X_adversarial = X.copy()

    weights = model.coef_[0]

    # Move feature influence toward benign class
    if target_class == 0:
        feature_influence = -weights
    else:
        feature_influence = weights

    # Rank features according to influence
    important_features = np.argsort(
        np.abs(feature_influence)
    )[::-1]

    selected_features = (
        important_features[
            :max_features
        ]
    )

    # Modify selected features
    for feature_index in selected_features:

        direction = np.sign(
            feature_influence[
                feature_index
            ]
        )

        if direction == 0:
            continue

        X_adversarial[
            :,
            feature_index
        ] += (
            theta * direction
        )

    return X_adversarial


if __name__ == "__main__":

    print("JSMA Attack")
    print("-----------")
    print(
        "JSMA is used only during testing."
    )
    print(
        "JSMA is an UNSEEN attack."
    )
