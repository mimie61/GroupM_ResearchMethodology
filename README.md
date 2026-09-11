# Adversarial Training for Cross-Attack Robustness in ML-Based Intrusion Detection

## Group Information

**Group:** M

### Group Members

| Name | Student ID |
|---|---|
| Zuhairra Shamimie Binti Zainal | 52215124219 |
| Wan Nur Aisyah Binti Long Mahmood | 52215124433 |
| Wan Johanna Eriesha Binti Mior Mohd Nasir | 52215124471 |
| Muhammad Amirun Mustaqim Bin Mohd Affendi | 52215125801 |

**Course:** IDB30102 Research Methodology  
**Programme:** Bachelor of Cybersecurity Technology with Honours  
**Research Area:** Adversarial Machine Learning in Cybersecurity  
**Application Area:** Machine-Learning-Based Intrusion Detection

---

# 1. Research Overview

## 1.1 Research Problem

Machine-learning-based Intrusion Detection Systems (IDSs) can be vulnerable to adversarial attacks that deliberately modify input data to cause incorrect classification.

Existing adversarial machine learning defenses are often evaluated against a limited number of attack techniques. Therefore, it remains unclear whether a defense trained against one adversarial attack can maintain its effectiveness against other attacks that were not included during training.

This research focuses on investigating whether an adversarially trained Logistic Regression-based IDS can maintain robustness when exposed to different adversarial attacks.

## 1.2 Research Aim

The aim of this research is:

> To investigate whether an adversarially trained Logistic Regression-based IDS, trained using FGSM, can maintain robustness against other adversarial attacks (JSMA and DeepFool).

## 1.3 Research Objectives

The research objectives are:

1. To establish the baseline performance of a Logistic Regression-based IDS on CIC-IDS2017 under FGSM, JSMA, and DeepFool attacks.

2. To implement an adversarially trained Logistic Regression IDS using FGSM-generated adversarial examples.

3. To assess the cross-attack robustness of the FGSM-trained defense against FGSM, JSMA, and DeepFool.

---

# 2. Proposed Solution

The proposed research uses **FGSM-based adversarial training** as the defense mechanism.

A Logistic Regression model will first be trained using clean CIC-IDS2017 training data to establish a baseline.

FGSM adversarial examples will then be generated from the training data and combined with the clean training samples to train a defended Logistic Regression model.

The defended model will subsequently be evaluated against:

- FGSM
- JSMA
- DeepFool

FGSM is considered the **seen attack** because it is used during adversarial training.

JSMA and DeepFool are considered **unseen attacks** because they are not included during adversarial training.

The comparison will determine whether robustness learned from FGSM can extend to adversarial attacks that were not used during training.

---

# 3. Research Methodology

## 3.1 Research Approach

The research adopts an **Experimental Research Approach**.

The experiment compares a baseline Logistic Regression IDS with an FGSM-trained Logistic Regression IDS under clean and adversarial testing conditions.

The experiment will use the CIC-IDS2017 dataset and evaluate model performance using Accuracy, Precision, Recall, F1-score, and Attack Success Rate (ASR).

## 3.2 Development Model

The proposed technical components will follow the **Iterative and Incremental Development Model**.

The technical components will be developed incrementally, including:

1. Dataset loading and preprocessing
2. Logistic Regression baseline
3. FGSM adversarial attack generation
4. FGSM adversarial training
5. JSMA and DeepFool testing components
6. Testing and evaluation

Each component can be refined during development before being integrated into the proposed experimental workflow.

---

# 4. Research Dataset

## CIC-IDS2017

The proposed experiment uses the **CIC-IDS2017** dataset.

CIC-IDS2017 contains labelled benign and malicious network traffic and provides flow-level network features suitable for supervised machine-learning-based intrusion detection research.

The dataset will be preprocessed before model training.

### Proposed preprocessing

- Remove invalid or missing values
- Remove duplicate records
- Separate features and labels
- Convert labels into binary classes
  - `0 = Benign`
  - `1 = Malicious`
- Convert categorical information into numerical form where required
- Perform a stratified **70:30 train-test split**
- Apply **MinMaxScaler**
- Fit the scaler using training data only

The complete CIC-IDS2017 dataset is not uploaded to this repository. Dataset information and its source are provided in:

`05_Data_or_Sample_Input/`

---

# 5. Experimental Design

The experiment consists of seven testing configurations.

| No. | Model | Test Data / Attack | Attack Status |
|---|---|---|---|
| 1 | Baseline Logistic Regression | Clean | N/A |
| 2 | Baseline Logistic Regression | FGSM | Seen |
| 3 | Baseline Logistic Regression | JSMA | Unseen |
| 4 | Baseline Logistic Regression | DeepFool | Unseen |
| 5 | FGSM-trained Logistic Regression | FGSM | Seen |
| 6 | FGSM-trained Logistic Regression | JSMA | Unseen |
| 7 | FGSM-trained Logistic Regression | DeepFool | Unseen |

### Training Configuration

The baseline model is trained using clean training data.

The defended model is trained using:

> Clean training data + FGSM adversarial training examples

FGSM is the only adversarial attack used during adversarial training.

JSMA and DeepFool are reserved for testing.

---

# 6. Evaluation Plan

The proposed models will be evaluated using the following metrics:

### Accuracy

Measures the proportion of correctly classified samples.

### Precision

Measures the proportion of samples predicted as a class that are correctly classified.

### Recall

Measures the ability of the model to correctly identify the relevant class.

### F1-score

Provides a combined measure of precision and recall.

### Attack Success Rate (ASR)

ASR measures the proportion of malicious adversarial samples that are classified as benign.

For this research:

> Successful attack = Malicious traffic → Benign prediction

A lower ASR indicates stronger resistance to successful adversarial evasion.

The evaluation will compare the baseline and FGSM-trained models to determine:

- Whether adversarial training improves robustness against FGSM.
- Whether robustness against FGSM extends to JSMA.
- Whether robustness against FGSM extends to DeepFool.
- Whether the defense mainly provides seen-attack robustness or demonstrates cross-attack robustness.

No final experimental results are reported in this repository because the experiment is still at the proposal stage.

---

# 7. Architecture and Experimental Flow

The proposed experimental workflow is:

```text
CIC-IDS2017 Dataset
        |
        v
Data Cleaning & Preprocessing
        |
        v
70:30 Stratified Train-Test Split
        |
        v
MinMaxScaler
        |
        +--------------------------+
        |                          |
        v                          v
Baseline Logistic          FGSM Adversarial
Regression                  Training
        |                          |
        |                          v
        |                   FGSM-trained
        |                   Logistic Regression
        |                          |
        +-------------+------------+
                      |
                      v
              Adversarial Testing
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
        FGSM        JSMA      DeepFool
        Seen       Unseen       Unseen
          |           |           |
          +-----------+-----------+
                      |
                      v
                Evaluation
                      |
          +-----------+-----------+
          |      |      |      |  |
          v      v      v      v  v
       Accuracy Precision Recall F1 ASR
                      |
                      v
          Cross-Attack Robustness
                 Analysis
