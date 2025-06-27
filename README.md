## Credit Scoring Business Understanding

### How Basel II Influences Model Design
The Basel II Accord encourages financial institutions to quantify and manage credit risk through internal models. This necessitates building interpretable, auditable, and reproducible machine learning systems. Our model must be explainable for regulatory compliance and decision transparency.

### Why a Proxy Variable is Needed
Since we lack a direct default indicator, we derive a proxy target using Recency, Frequency, and Monetary metrics (RFM). While this proxy allows model training, it introduces a risk of mislabeling, which can impact real-world loan decisions. Careful validation and explainability are critical.

### Interpretable vs Complex Models in Finance
- **Simple models** (e.g. Logistic Regression with WoE) are preferred by regulators due to transparency and explainability.
- **Complex models** (e.g. Random Forest, XGBoost) may provide better performance but require tools like SHAP for justification in financial decisions.
