# Credit Risk Probability Model

This project aims to predict the probability that a user is high-risk based on alternative transaction data. It simulates a real-world credit scoring system using synthetic labels, feature engineering, model training, and deployment.

### How Basel II Influences Model Design

Basel II emphasizes risk-based capital requirements, obligating financial institutions to hold capital reserves proportionate to the credit risk of their portfolios. This makes **risk quantification and transparency** essential. The Internal Ratings-Based (IRB) approach under Basel II allows banks to develop internal models for estimating Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). Consequently, our model must be **interpretable, documented, and auditable** to meet regulatory expectations.

### Why a Proxy Variable is Needed

Our dataset lacks a direct "default" label. To train a credit scoring model, we define a **proxy variable** by segmenting customers using behavioral features like Recency, Frequency, and Monetary value (RFM). This approach approximates default risk by labeling low-engagement clusters as "high-risk." While necessary, this proxy introduces potential **bias and regulatory risk** if misinterpreted. Validating this approach and disclosing limitations is essential.

### Interpretable vs Complex Models in Finance

- **Simple Models** (e.g., Logistic Regression with Weight of Evidence encoding) offer transparency, ease of audit, and alignment with Basel II compliance standards.
- **Complex Models** (e.g., Random Forest, Gradient Boosting) offer superior predictive performance but require advanced interpretability tools (e.g., SHAP) to justify use in regulated environments. These models must be carefully monitored for fairness and bias.

---


## 🚀 Project Goals

- Apply predictive modeling to identify high-risk customers
- Create proxy labels via unsupervised clustering
- Build robust pipelines for transformation and training
- Log models with MLflow
- Deploy using FastAPI, Docker, and GitHub Actions

## 📁 Project Structure

```
.
├── data/
│   ├── raw/               # Raw input files
│   └── processed/         # Cleaned/engineered data
├── src/
│   ├── data_processing.py # Feature engineering pipeline
│   ├── proxy_label.py     # Label generation using RFM + KMeans
│   ├── train.py           # Model training with MLflow
│   └── api/               # FastAPI prediction service
├── tests/
│   └── test_data_processing.py  # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/
    └── workflows/
        └── ci.yml         # GitHub Actions for lint & test
```

## 🧠 Key Steps

### 1. Business Understanding
- Uses Basel II internal risk model framework
- No ground truth defaults — uses proxy labeling

### 2. EDA
- No missing values
- Mixed data types
- Detected negatives in Amount (possible refunds)

### 3. Proxy Target Creation
- Created `is_high_risk` label using:
  - Recency, Frequency, Monetary (RFM)
  - KMeans clustering
- Label used for classification

### 4. Modeling
- `LogisticRegression` for interpretability
- `GradientBoostingClassifier` for performance
- MLflow used for model tracking

### 5. Deployment
- FastAPI `/predict` endpoint
- Loads model from MLflow Registry
- Dockerized
- GitHub CI pipeline checks lint and tests

## 🔗 GitHub Repo

[https://github.com/kirubhel/credit-risk-model](https://github.com/kirubhel/credit-risk-model)

## 🧪 Run Tests

```bash
pytest
```

## 📦 Run API Locally

```bash
docker-compose up --build
```

Then open: `http://localhost:8000/docs` for Swagger UI.

---

Created by **Kirubel Gizaw** for 10 Academy B5W5 Challenge.
