# 08_testing_evaluation.py
# Preliminary testing and evaluation

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Calculate classification metrics.

    Metrics:
        Accuracy
        Precision
        Recall
        F1-score
    """

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


def calculate_attack_success_rate(
    model,
    X_adversarial,
    y_true
):
    """
    Calculate Attack Success Rate (ASR).

    For this IDS research, an attack is considered
    successful when malicious traffic is classified
    as benign.

    Therefore:

        True label     = 1 (malicious)
        Prediction     = 0 (benign)

    ASR =
        Successful malicious-to-benign evasions
        /
        Total malicious adversarial samples
        × 100
    """

    predictions = model.predict(
        X_adversarial
    )

    # Select only originally malicious samples
    malicious_samples = (
        y_true == 1
    )

    total_malicious = (
        malicious_samples.sum()
    )

    if total_malicious == 0:
        return 0.0

    # Successful evasion:
    # malicious → benign
    successful_attacks = (
        (y_true == 1)
        &
        (predictions == 0)
    )

    successful_count = (
        successful_attacks.sum()
    )

    asr = (
        successful_count
        / total_malicious
        * 100
    )

    return asr


def display_results(
    model_name,
    attack_name,
    metrics,
    asr=None
):
    """
    Display evaluation results.
    """

    print("\n================================")
    print("Model :", model_name)
    print("Attack:", attack_name)
    print("================================")

    print(
        "Accuracy :",
        round(
            metrics["Accuracy"],
            4
        )
    )

    print(
        "Precision:",
        round(
            metrics["Precision"],
            4
        )
    )

    print(
        "Recall   :",
        round(
            metrics["Recall"],
            4
        )
    )

    print(
        "F1-score :",
        round(
            metrics["F1-score"],
            4
        )
    )

    if asr is not None:

        print(
            "ASR (%)  :",
            round(asr, 2)
        )


if __name__ == "__main__":

    print(
        "Testing and Evaluation Module"
    )

    print("\nEvaluation Metrics:")
    print("- Accuracy")
    print("- Precision")
    print("- Recall")
    print("- F1-score")
    print("- Attack Success Rate (ASR)")

    print("\nTesting Configurations:")
    print("1. Baseline LR + Clean")
    print("2. Baseline LR + FGSM")
    print("3. Baseline LR + JSMA")
    print("4. Baseline LR + DeepFool")
    print("5. FGSM-trained LR + FGSM")
    print("6. FGSM-trained LR + JSMA")
    print("7. FGSM-trained LR + DeepFool")
