# Credit Risk Probability Model for Alternative Data

## Project Overview

This project develops an end-to-end credit risk scoring solution for Bati Bank using transaction data from the Xente eCommerce platform.

---

## Dataset

This project uses the Xente Challenge dataset from Kaggle.

Source:
https://www.kaggle.com/competitions/xente-fraud-detection

The dataset contains transaction-level information collected from the Xente eCommerce platform, including customer identifiers, account information, transaction amounts, product details, timestamps, pricing strategies, and fraud indicators.

Access Notes:

- The dataset must be downloaded separately from Kaggle.
- Raw datasets are excluded from version control using `.gitignore`.
- Users reproducing this work must place the dataset in:

```text
data/raw/data.csv
```

---

## Project Structure

```text
credit-risk-model/
├── .github/
├── data/
├── notebooks/
├── src/
├── tests/
├── README.md
```

---

## Credit Scoring Business Understanding

### How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

Basel II requires financial institutions to use transparent and well-documented risk assessment methods. Since lending decisions directly affect capital allocation and regulatory compliance, models must be explainable and auditable. Interpretable models allow regulators, auditors, and risk managers to understand how predictions are generated and ensure decisions can be justified.

### Without a direct default label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The dataset does not contain actual loan default outcomes. Therefore, a proxy variable must be created to approximate credit risk. Behavioral indicators such as Recency, Frequency, and Monetary value can be used to identify potentially risky customers. However, proxy targets are assumptions rather than true defaults, creating risks of misclassification, biased decisions, and reduced model reliability.

### What are the key trade-offs between a simple interpretable model and a high-performance model?

Logistic Regression combined with Weight of Evidence provides strong interpretability and easier regulatory compliance. Gradient Boosting models often achieve higher predictive performance but are more difficult to explain and audit. Financial institutions must balance predictive accuracy with transparency and governance requirements.