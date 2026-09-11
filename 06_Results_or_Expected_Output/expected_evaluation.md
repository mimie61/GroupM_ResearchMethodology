# Expected Evaluation

## Evaluation Objective

The proposed experiment will compare a baseline Logistic Regression (LR) model with an FGSM-trained Logistic Regression model to investigate cross-attack robustness in an ML-based intrusion detection system.

## Testing Configurations

Seven testing configurations will be evaluated:

| No. | Model | Test Data / Attack | Attack Status |
|---|---|---|---|
| 1 | Baseline LR | Clean test data | N/A |
| 2 | Baseline LR | FGSM | Seen |
| 3 | Baseline LR | JSMA | Unseen |
| 4 | Baseline LR | DeepFool | Unseen |
| 5 | FGSM-trained LR | FGSM | Seen |
| 6 | FGSM-trained LR | JSMA | Unseen |
| 7 | FGSM-trained LR | DeepFool | Unseen |

## Evaluation Metrics

The proposed evaluation will use the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Attack Success Rate (ASR)

ASR will be used to measure the proportion of malicious adversarial samples that are classified as benign by the IDS. A lower ASR indicates stronger resistance to successful adversarial evasion.

## Expected Outcomes

The baseline Logistic Regression model is expected to show reduced performance when exposed to adversarial examples.

The FGSM-trained Logistic Regression model is expected to demonstrate improved robustness against FGSM because FGSM adversarial examples are included during adversarial training.

The performance of the FGSM-trained model against JSMA and DeepFool will be examined to determine whether robustness learned from FGSM can extend to unseen adversarial attacks.

The comparison will therefore determine whether FGSM-based adversarial training provides robustness only against the seen attack or whether some level of cross-attack robustness can be observed against unseen attacks.

No numerical results are reported at the proposal stage because the proposed experiment has not yet been conducted.