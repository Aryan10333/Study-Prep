# Module 05: Classical NLP Models & Baseline Classifiers

This study guide covers classical machine learning models for NLP, Bayes' Theorem, Naive Bayes (MultinomialNB & BernoulliNB), Laplace Smoothing, Logistic Regression for text, Linear SVM, Conditional Random Fields (CRF), step-by-step Naive Bayes hand-calculations, Scikit-Learn pipelines, failure modes, and interview flashcards.

> **Notebook Companion**: [05_classical_nlp_models_and_baselines.ipynb](file:///d:/Study/Prep/machine-learning-prep/nlp/05_classical_nlp_models_and_baselines.ipynb)

---

## 1. Why Classical NLP Models Matter in Production

While modern Transformer LLMs offer state-of-the-art accuracy, classical statistical machine learning models (Naive Bayes, Logistic Regression, Linear SVM) remain indispensable in enterprise software architectures for key engineering reasons:

1. **Ultra-Low Latency SLAs**: Inference execution takes $<2\text{ms}$ on CPU, compared to $100\text{ms}-1000\text{ms}+$ for deep LLM transformer calls.
2. **Zero Compute & API Cost**: Operates efficiently on standard CPU web servers without expensive GPU clusters.
3. **High Throughput Batching**: Classifies millions of incoming stream documents (e.g. log filtering, spam detection) per second.
4. **Strong Baseline Metric**: Establishes a fast, deterministic accuracy benchmark before deploying complex neural networks.

---

## 2. Naive Bayes Classifier Mechanics

Naive Bayes applies **Bayes' Theorem** with the "naive" conditional independence assumption that features (word occurrences) are independent given the class $c$.

### 1. Bayes' Theorem for Text Classification:
$$P(c | d) = \frac{P(d | c) P(c)}{P(d)} = \frac{P(w_1, w_2, \dots, w_n | c) P(c)}{P(w_1, w_2, \dots, w_n)}$$

### 2. Naive Independence Assumption:
$$P(w_1, w_2, \dots, w_n | c) = \prod_{i=1}^n P(w_i | c)$$

$$\hat{c} = \arg\max_{c \in C} P(c) \prod_{i=1}^n P(w_i | c)$$

In log-space (to prevent floating-point underflow):

$$\hat{c} = \arg\max_{c \in C} \left[ \log P(c) + \sum_{i=1}^n \log P(w_i | c) \right]$$

### 3. Multinomial Naive Bayes & Laplace Smoothing ($\alpha=1$)
If a word $w_i$ in the test sentence never appeared in class $c$ during training, raw frequency probability yields $P(w_i | c) = 0$, causing the entire product $\prod P(w_i | c) = 0$.

To solve this, **Laplace Add-1 Smoothing** adds a constant $\alpha = 1$ to the numerator and $\alpha \cdot |V|$ to the denominator:

$$P(w_i | c) = \frac{N_{c,i} + \alpha}{N_c + \alpha \cdot |V|}$$

Where:
- $N_{c,i}$ is the total count of word $w_i$ in class $c$.
- $N_c$ is the total token count across all documents in class $c$.
- $|V|$ is the total vocabulary size.

---

## 3. Discriminative Linear Models: Logistic Regression & Linear SVM

```text
Model Paradigm       Loss Function                              Decision Boundary              Best Use Case
--------------------------------------------------------------------------------------------------------------------------------------
Naive Bayes (Generative) Joint Likelihood P(X, Y)                Probabilistic Linear           Ultra-fast text baseline
Logistic Regression  Log-Loss / Cross-Entropy                    Linear Sigmoid Hyperplane      Calibrated probability classification
Linear SVM           Hinge Loss: max(0, 1 - y_i (w^T x_i + b)) Max-Margin Linear Hyperplane   High-dimensional sparse text classification
CRF                  Global Sequence Log-Likelihood              Linear-Chain Graphical Model   Named Entity Recognition (NER)
```

### 1. Logistic Regression for Text
Learns weights $w$ that maximize the conditional log-likelihood of class $y \in \{0, 1\}$ given sparse feature vector $x$:

$$P(y=1 | x) = \sigma(w^\top x + b) = \frac{1}{1 + \exp(-(w^\top x + b))}$$

- **L1 Regularization (Lasso)**: Forces irrelevant feature weights to exactly zero, performing automated feature selection.
- **L2 Regularization (Ridge)**: Prevents overfitting on rare sparse terms by shrinking weights toward zero.

### 2. Linear Support Vector Machine (LinearSVC)
Finds the optimal separating hyperplane $w^\top x + b = 0$ that maximizes the margin distance between text classes:

$$\min_{w} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \max\left(0, 1 - y_i (w^\top x_i + b)\right)$$

Linear SVMs perform exceptionally well on high-dimensional sparse TF-IDF feature spaces ($|V| \ge 50,000$).

---

## 4. Sequence Tagging: Conditional Random Fields (CRF)

Unlike independent token classifiers, a **Conditional Random Field (CRF)** is a discriminative undirected graphical model that models tag transitions across token sequences $Y = (y_1, \dots, y_T)$ given input text sequence $X = (x_1, \dots, x_T)$.

$$P(Y | X) = \frac{1}{Z(X)} \exp\left( \sum_{t=1}^T \sum_k \lambda_k f_k(y_t, y_{t-1}, X, t) \right)$$

CRFs prevent invalid tag sequences in Named Entity Recognition (e.g. prohibiting an `I-PER` tag from following an `O` tag without a preceding `B-PER` tag).

---

## 5. Step-by-Step Hand Calculation Example (Andrew Ng Style)

Suppose we have a tiny training dataset of $D = 3$ documents for binary Sentiment Analysis:
- **Positive Class ($c = +$)**:
  - Doc 1: `"great product"`
  - Doc 2: `"great support"`
- **Negative Class ($c = -$)**:
  - Doc 3: `"bad product"`

We evaluate Test Document $d_{\text{test}} = \text{"great service"}$.

### 1. Prior Probabilities $P(c)$:
- $P(+) = 2 / 3 \approx \mathbf{0.6667}$
- $P(-) = 1 / 3 \approx \mathbf{0.3333}$

### 2. Extract Vocabulary ($|V| = 5$):
$$V = \{\text{"great"}, \text{"product"}, \text{"support"}, \text{"bad"}, \text{"service"}\} \implies |V| = \mathbf{5}$$

### 3. Total Tokens per Class ($N_c$):
- $N_+ = 2 + 2 = \mathbf{4\text{ tokens}}$ (`"great"`, `"product"`, `"great"`, `"support"`)
- $N_- = 2 = \mathbf{2\text{ tokens}}$ (`"bad"`, `"product"`)

### 4. Compute Likelihoods with Laplace Smoothing ($\alpha=1$):
$$P(w_i | c) = \frac{N_{c,i} + 1}{N_c + 5}$$

- **For Positive Class ($c = +$, $N_+ + 5 = 9$):**
  - $P(\text{"great"} | +) = (2 + 1) / 9 = 3 / 9 = \mathbf{0.3333}$
  - $P(\text{"service"} | +) = (0 + 1) / 9 = 1 / 9 = \mathbf{0.1111}$

- **For Negative Class ($c = -$, $N_- + 5 = 7$):**
  - $P(\text{"great"} | -) = (0 + 1) / 7 = 1 / 7 = \mathbf{0.1429}$
  - $P(\text{"service"} | -) = (0 + 1) / 7 = 1 / 7 = \mathbf{0.1429}$

### 5. Compute Posterior Probabilities $P(c) \times \prod P(w_i | c)$:
- **Positive Class Score**:

  $$\text{Score}(+) = P(+) \times P(\text{"great"} | +) \times P(\text{"service"} | +) = \left(\frac{2}{3}\right) \times \left(\frac{3}{9}\right) \times \left(\frac{1}{9}\right) = \frac{6}{243} \approx \mathbf{0.02469}$$

- **Negative Class Score**:

  $$\text{Score}(-) = P(-) \times P(\text{"great"} | -) \times P(\text{"service"} | -) = \left(\frac{1}{3}\right) \times \left(\frac{1}{7}\right) \times \left(\frac{1}{7}\right) = \frac{1}{147} \approx \mathbf{0.00680}$$

### 6. Decision Rule:
$$\text{Score}(+) = 0.02469 > \text{Score}(-) = 0.00680 \implies \mathbf{\hat{y} = \text{Positive}}$$

---

## 6. Production Python Code Implementation

```python
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Real-World Support Ticket Routing Corpus
X_train = [
    "Database connection timeout on port 5432 postgresql replica",
    "SQL query execution failed due to lock contention",
    "Billing credit card payment authorization declined 402",
    "Invoice subscription refund request for billing cycle",
    "Database disk space full transaction log cleanup",
    "Monthly billing invoice charges unexpected amount"
]
y_train = ["Infrastructure", "Infrastructure", "Billing", "Billing", "Infrastructure", "Billing"]

# 2. Build Production Pipeline (TF-IDF + Logistic Regression)
clf_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)),
    ("classifier", LogisticRegression(C=1.0, max_iter=100))
])

# 3. Fit Pipeline & Save Model Artifact to Hidden Folder
clf_pipeline.fit(X_train, y_train)

os.makedirs(".model_cache", exist_ok=True)
model_path = os.path.join(".model_cache", "ticket_classifier.pkl")
joblib.dump(clf_pipeline, model_path)

print(f"=== Trained Pipeline Artifact Saved to Hidden Folder: {model_path} ===")

# 4. Inference Execution
X_test = ["PostgreSQL database primary server crash", "Credit card failed billing charge"]
predictions = clf_pipeline.predict(X_test)
probabilities = clf_pipeline.predict_proba(X_test)

print("\n=== Model Inference Results ===")
for text, pred, proba in zip(X_test, predictions, probabilities):
    max_prob = max(proba)
    print(f"Text: '{text}'")
    print(f"  Predicted Category: {pred} (Confidence: {max_prob*100:.1f}%)\n")
```

> [!NOTE]
> **Production Baseline Alert:**
> - Scikit-Learn `Pipeline` encapsulates feature extraction and classification into a single atomic artifact. Always save the entire `Pipeline` object via `joblib.dump` to prevent vectorizer feature index mismatch bugs during API serving.

---

## 7. Production Failure Modes & Selection Rules

### Production Failure Modes:
1. **Class Imbalance Skew**: In a dataset with 99% Non-Spam and 1% Spam, Naive Bayes and Logistic Regression will predict Non-Spam for almost all samples to maximize accuracy.
   - *Remediation*: Use `class_weight="balanced"` in Logistic Regression or evaluate Macro F1 / Precision-Recall AUC instead of raw accuracy.
2. **Feature Mismatch in Pipelines**: Passing raw text into a model file saved separately from its `TfidfVectorizer` causes index misalignment and silent wrong predictions.
   - *Remediation*: Always encapsulate vectorizer and classifier in a single Scikit-Learn `Pipeline`.

---

## 8. Master Interview Flashcards & Questions

#### Q1: Why does Naive Bayes perform surprisingly well in practice despite its "naive" independence assumption?
- **Answer:** Naive Bayes relies on word independence $P(w_1, w_2 | c) = P(w_1 | c) P(w_2 | c)$. While word frequencies are strongly correlated in real language, classification decision boundaries depend only on the relative argmax rank between classes rather than exact probability estimation. As long as the strongest evidence terms point in the correct direction, rank order remains intact.

#### Q2: What is the purpose of Laplace Smoothing in Naive Bayes?
- **Answer:** If a test word $w_{\text{new}}$ never appeared in class $c$ during training, its empirical frequency probability is $P(w_{\text{new}} | c) = 0$. Because Naive Bayes multiplies feature probabilities together, a single zero causes the entire class score to become zero. Laplace Add-1 Smoothing adds $\alpha = 1$ to the numerator and $\alpha \cdot |V|$ to the denominator, guaranteeing non-zero probabilities for unseen words.

#### Q3: Compare Naive Bayes (Generative) vs Logistic Regression (Discriminative) for text classification.
- **Answer:** Naive Bayes is a generative model that learns the joint probability $P(X, Y) = P(X|Y)P(Y)$. It trains faster and works well on tiny datasets, but suffers if features are highly correlated. Logistic Regression is a discriminative model that directly learns conditional probability $P(Y|X) = \sigma(w^\top x + b)$. It scales better to large datasets, handles correlated n-gram features, and produces well-calibrated probabilities.
