# Credit Risk Probability Model for Alternative Data

## Project Overview

This project was completed as part of the Bati Bank Credit Scoring Challenge.

Bati Bank aims to introduce a Buy Now, Pay Later (BNPL) service by leveraging transaction data from Xente's mobile money platform. Since traditional credit bureau information is unavailable for many customers, alternative behavioral data must be used to estimate customer credit risk.

The objective of this project is to build an end-to-end machine learning system that:

* Creates a proxy credit risk target from transaction behavior
* Engineers customer-level features from transaction history
* Trains and evaluates credit risk prediction models
* Tracks experiments using MLflow
* Deploys the selected model through a FastAPI application
* Provides containerization support through Docker
* Implements CI/CD through GitHub Actions

---

## Business Context

Financial institutions must comply with Basel II principles, which require risk models to be interpretable, measurable, and auditable.

Because a true loan-default label is unavailable, this project creates a proxy target variable using customer behavioral segmentation based on:

* Recency
* Frequency
* Monetary value (RFM)

Customers exhibiting riskier transaction behavior are identified through clustering and assigned a high-risk label.

The resulting model can support:

* Loan approval decisions
* Credit limit assignment
* Risk-based pricing
* Customer monitoring

---

## Dataset

This project uses the Xente transaction dataset provided for the Bati Bank Credit Scoring Challenge.

The dataset contains:

* Transaction identifiers
* Customer identifiers
* Account information
* Product categories
* Channel information
* Transaction timestamps
* Transaction amounts
* Fraud indicators

The data represents mobile money activity and is used to derive customer behavioral profiles.

Access to the dataset is provided through the challenge platform and may require registration.

---

## Repository Structure

```text
credit-risk-model/

├── data/
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   ├── proxy_target_engineering.ipynb
│   └── model_training.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
│
├── tests/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Task 1: Business Understanding

Completed activities:

* Defined business objective
* Evaluated Basel II implications
* Assessed proxy target approaches
* Established repository structure
* Configured Git workflow

---

## Task 2: Exploratory Data Analysis

Completed analyses:

* Dataset overview
* Data type inspection
* Summary statistics
* Missing value analysis
* Duplicate analysis
* Outlier detection
* Numerical feature distributions
* Categorical feature distributions
* Correlation analysis

Key findings:

* Transaction amount contains significant outliers
* Several categorical variables have high cardinality
* Missing values are limited
* Transaction behavior varies significantly between customers

---

## Task 3: Feature Engineering

Implemented features:

### Aggregate Features

* TotalTransactionAmount
* AverageTransactionAmount
* TransactionCount
* StdTransactionAmount

### Time Features

* TransactionHour
* TransactionDay
* TransactionMonth
* TransactionYear

### Transformations

* Missing-value imputation
* One-hot encoding
* Feature scaling

---

## Task 4: Proxy Target Engineering

An RFM framework was used to generate a proxy credit risk target.

### RFM Components

Recency:

* Days since last transaction

Frequency:

* Number of transactions

Monetary:

* Total transaction amount

### Clustering

K-Means clustering was applied to customer RFM profiles.

Customers belonging to the riskiest cluster were labeled:

```text
is_high_risk = 1
```

All other customers were labeled:

```text
is_high_risk = 0
```

This generated the target variable used for model training.

---

## Task 5: Model Training and Experiment Tracking

Models evaluated:

### Logistic Regression

* Accuracy: 0.6729
* Precision: 0.5872
* Recall: 0.4825
* F1 Score: 0.5298
* ROC-AUC: 0.7284

### Random Forest

* Accuracy: 0.6943
* Precision: 0.5882
* Recall: 0.6643
* F1 Score: 0.6240
* ROC-AUC: 0.7548

### Selected Model

Random Forest achieved the highest ROC-AUC and strongest recall performance and was selected as the final model.

MLflow was used to:

* Track experiments
* Log metrics
* Store model artifacts
* Compare model performance

---

## Task 6: Deployment

### FastAPI

Run locally:

```bash
uvicorn src.api.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### Example Request

```json
{
  "TotalTransactionAmount": 50000,
  "AverageTransactionAmount": 2500,
  "TransactionCount": 20,
  "StdTransactionAmount": 1000,
  "FraudResult": 0
}
```

### Example Response

```json
{
  "prediction": 0,
  "probability": 0.42
}
```

---

## Docker

Build image:

```bash
docker build -t credit-risk-api .
```

Run container:

```bash
docker run -p 8000:8000 credit-risk-api
```

---

## CI/CD

GitHub Actions automatically:

* Installs dependencies
* Executes tests
* Validates pull requests

Workflow file:

```text
.github/workflows/ci.yml
```

---

## Author

Arsema Esayas

Data Science & Credit Risk Modeling Project

Bati Bank Credit Scoring Challenge
