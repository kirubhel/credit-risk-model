## Credit Scoring Business Understanding

### How Basel II Influences Model Design

Basel II emphasizes risk-based capital requirements, obligating financial institutions to hold capital reserves proportionate to the credit risk of their portfolios. This makes **risk quantification and transparency** essential. The Internal Ratings-Based (IRB) approach under Basel II allows banks to develop internal models for estimating Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). Consequently, our model must be **interpretable, documented, and auditable** to meet regulatory expectations.

### Why a Proxy Variable is Needed

Our dataset lacks a direct "default" label. To train a credit scoring model, we define a **proxy variable** by segmenting customers using behavioral features like Recency, Frequency, and Monetary value (RFM). This approach approximates default risk by labeling low-engagement clusters as "high-risk." While necessary, this proxy introduces potential **bias and regulatory risk** if misinterpreted. Validating this approach and disclosing limitations is essential.

### Interpretable vs Complex Models in Finance

- **Simple Models** (e.g., Logistic Regression with Weight of Evidence encoding) offer transparency, ease of audit, and alignment with Basel II compliance standards.
- **Complex Models** (e.g., Random Forest, Gradient Boosting) offer superior predictive performance but require advanced interpretability tools (e.g., SHAP) to justify use in regulated environments. These models must be carefully monitored for fairness and bias.

---

## Project Structure

credit-risk-model/
├── data/
│ ├── raw/
│ │ ├── data.csv
│ │ ├── data.xlsx
│ │ ├── Xente_Variable_Definitions.csv
│ │ └── Xente_Variable_Definitions.xlsx
│ └── processed/
├── notebooks/
├── src/
│ ├── init.py
│ ├── data_processing.py
│ ├── train.py
│ ├── predict.py
│ └── api/
│ ├── main.py
│ └── pydantic_models.py
├── tests/
│ └── test_data_processing.py
├── .github/
│ └── workflows/
│ └── ci.yml
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .gitignore
└── README.md