# Spam Email Detection: Introducing Linear Regression & Text Classification

A minimal, clean, and self-contained Machine Learning project in Python comparing **Linear Regression**, **Logistic Regression**, and **Multinomial Naive Bayes** for spam email classification.

---

## 📌 Project Overview

This project is a clean, practical Machine Learning study that explores why **Linear Regression** conceptually fails when applied to binary classification, introduces **Logistic Regression** as the proper probabilistic linear model, and compares both against the industry-standard text baseline: **Multinomial Naive Bayes**.

---

## 🚀 Quickstart

### 1. Requirements
- Python 3.9+
- Dependencies: `pandas`, `numpy`, `scikit-learn`

```bash
pip install -r requirements.txt
```

### 2. Run the Script
```bash
python spam_detection.py
python spam_detection.py --interactive
```

### 3. Open in Jupyter Notebook
You can also explore `spam_detection.ipynb` step-by-step in VS Code or Jupyter Notebook:
```bash
jupyter notebook spam_detection.ipynb
```

---

## 📊 Dataset & Preprocessing

- **Dataset**: Standard [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,572 messages: 4,825 Ham, 747 Spam).
- **Auto-Download & Caching**: Auto-downloads on first run and caches locally to `sms_spam_data.csv`. Includes a built-in offline fallback dataset.
- **Cleaning**: Lowercasing, punctuation/number stripping via regex `re.sub(r'[^a-z\s]', ' ', text)`, and whitespace normalization.
- **Split**: 80/20 Stratified train/test split (`random_state=42`) preserving class ratios (86.6% Ham / 13.4% Spam).
- **Vectorization**: `TfidfVectorizer(max_features=1500, stop_words='english')`.
  - **Data Leakage Prevention**: Vectorizer is fitted strictly on `X_train` and applied to `X_test`.

---

## 🔬 Model Comparison & Results

Evaluated on the held-out test set ($N = 1,115$ messages; 966 Ham, 149 Spam):

| Model | Accuracy | Precision | Recall | F1-Score | False Positives (Costly) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Linear Regression (Threshold 0.5)** | 97.04% | 91.43% | **85.91%** | 88.58% | **12** ⚠️ |
| **2. Logistic Regression (Sigmoid)** | **97.49%** | **99.19%** | 81.88% | 89.71% | **1** ✅ |
| **3. Multinomial Naive Bayes (Baseline)** | **97.49%** | 97.64% | 83.22% | **89.86%** | **3** ✅ |

### Confusion Matrix Breakdown:
- **Linear Regression**: $TN = 954, \mathbf{FP = 12}, FN = 21, TP = 128$
- **Logistic Regression**: $TN = 965, \mathbf{FP = 1}, FN = 27, TP = 122$
- **Multinomial Naive Bayes**: $TN = 963, \mathbf{FP = 3}, FN = 25, TP = 124$

---

## 🧠 Theoretical & Mathematical Foundations

### 1. Why Linear Regression Fails for Classification

When applying Ordinary Least Squares (OLS) to binary targets ($y \in \{0, 1\}$):

$$\hat{y} = Xw + b, \quad \text{Loss} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

1. **Unbounded Output Range (Violates Probability Axioms)**:
   - Output space is $\hat{y} \in (-\infty, +\infty)$.
   - In our empirical test ($N = 1,115$):
     - **Minimum prediction**: `-0.4091` ($P < 0$ for **42.42%** of test samples!)
     - **Maximum prediction**: `+1.8166` ($P > 1$ for **4.66%** of test samples!)
     - Over **47%** of predictions violate fundamental probability rules.
2. **MSE Over-penalizes Highly Confident Points ("Too Correct")**:
   - If an email contains overt spam terms and produces $w^T x = 3.0$ with true label $y = 1$:
     - Logistic Regression calculates $P = \sigma(3.0) \approx 0.95 \implies \text{Loss} \approx 0$.
     - Linear Regression incurs a squared penalty: $(1 - 3.0)^2 = 4.0$!
     - OLS treats this confident point as a huge error and tilts the hyperplane, misclassifying legitimate emails and producing **12 False Positives** instead of 1.
3. **Heteroscedasticity**:
   - For binary data, $\text{Var}(Y|X) = p(x)(1 - p(x))$, which depends on $X$. This violates the Gauss-Markov homoscedasticity assumption required for OLS optimality.

---

### 2. Logistic Regression: The Proper Linear Classifier

Instead of fitting an unconstrained plane to binary labels, Logistic Regression models the **log-odds (logit)**:

$$z = \ln\left(\frac{p}{1 - p}\right) = w^T x + b$$

Applying the **Sigmoid function** maps $z \in (-\infty, +\infty)$ into valid probabilities $p \in (0, 1)$:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

#### Log-Loss (Binary Cross-Entropy):
$$J(w) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

- **Convexity**: Guaranteed global minimum (no local optima).
- **Gradient**: $\frac{\partial J}{\partial w} = \frac{1}{N} X^T (\hat{p} - y)$, which gracefully scales with classification error without penalizing confident correct classifications.

---

### 3. Multinomial Naive Bayes: Text Classification Benchmark

Applies Bayes' Theorem with the **"Naive" Conditional Independence Assumption**:

$$P(c | d) \propto P(c) \prod_{i=1}^V P(w_i | c)^{f_i}$$

#### Laplace (Additive) Smoothing ($\alpha = 1.0$):
$$\hat{P}(w_i | c) = \frac{N_{c, i} + \alpha}{N_c + \alpha \cdot |V|}$$

- **Why it matters**: If a rare test token was never observed in the Spam training set, raw $P(w_i | \text{Spam}) = 0$, which would zero out the entire product regardless of other spam tokens. Laplace smoothing prevents this zero-probability collapse.

---

### 4. Why Precision is King in Spam Filtering

In spam detection, the cost of errors is heavily **asymmetric**:

| Error Type | Formula | Scenario | Impact |
| :--- | :--- | :--- | :--- |
| **False Positive (FP)** | Type I Error | Legitimate email (job offer, OTP, hospital alert) routed to **Spam** | **CATASTROPHIC** (User misses vital communication) |
| **False Negative (FN)** | Type II Error | Spam email reaches **Inbox** | **TRIVIAL** (User clicks delete in 1 second) |

$$\text{Precision} = \frac{TP}{TP + FP}$$

- **Takeaway**: Maximizing Precision directly drives **False Positives to near zero**.
- **Result**: Logistic Regression achieved **99.19% Precision** with only **1 False Positive**, making it the best production candidate.

---

## 🧪 Live Sample Prediction Demo

```python
from spam_detection import predict_message, build_feature_pipeline, train_logistic_regression, load_dataset, clean_text
from sklearn.model_selection import train_test_split

df = load_dataset()
df['cleaned_text'] = df['text'].apply(clean_text)
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
vec, X_tr, X_te = build_feature_pipeline(X_train, X_test)
model, _, _ = train_logistic_regression(X_tr, y_train, X_te)

result = predict_message("Claim your free $1,000 Walmart gift card now!", vec, model)
print(result['prediction'])
print(f"Probability: {result['score']:.2%}")
```

---

## 📁 Project Structure

```
├── spam_detection.py       # Single clean, runnable script with all 3 models & CLI
├── spam_detection.ipynb    # Interactive Jupyter Notebook for visual exploration
├── sms_spam_data.csv       # Cached SMS Spam Collection dataset (auto-generated)
├── requirements.txt        # Minimal dependencies (pandas, numpy, scikit-learn)
└── README.md               # Complete documentation & architectural guide
```
