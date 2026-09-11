# 08_testing_evaluation.py
# Preliminary testing and evaluation for the IDS

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using standard classification metrics.
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

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

    print("Model Evaluation")
    print("----------------")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-score :", f1)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


def calculate_attack_success_rate(
    original_predictions,
    adversarial_predictions
):
    """
    Calculate the Attack Success Rate (ASR).

    ASR represents the proportion of originally
    correct predictions that become incorrect
    after adversarial manipulation.
    """

    successful_attacks = (
        original_predictions != adversarial_predictions
    )

    asr = successful_attacks.mean()

    print("Attack Success Rate (ASR):", asr)

    return asr


if __name__ == "__main__":

    print("Testing and Evaluation Module")
    print("--------------------------------")
    print("The defended model will be evaluated against:")
    print("FGSM  - Seen Attack")
    print("JSMA  - Unseen Attack")
    print("DeepFool - Unseen Attack")
