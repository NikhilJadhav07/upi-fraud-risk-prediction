# UPI Transaction Fraud & Risk Prediction (India)

Plain machine-learning project (no API, no Docker, no sklearn Pipeline). Everything runs in one notebook in VS Code.

## Files
| File | What it is |
|---|---|
| `UPI_Fraud_Risk_Prediction.ipynb` | The whole project: cleaning, features, EDA, 3 models, cost-based threshold, risk levels, single-transaction scoring |
| `upi_transactions_india.csv` | 100,000 SYNTHETIC UPI-style transactions (see note below) |
| `generate_data.py` | Script that created the CSV (shows exactly how fraud was simulated) |
| `upi_fraud_model.pkl`, `feature_names.pkl`, `threshold_info.pkl` | Saved by the notebook (Step 9) |

## Run in VS Code
1. Open this folder in VS Code and install the **Python** and **Jupyter** extensions.
2. Terminal: `python -m venv venv`, activate it, then `pip install -r requirements.txt`
3. Open the notebook, pick the `venv` kernel (top right), click **Run All**.
4. To regenerate the data first: `python generate_data.py`

## Important: the data is synthetic
Real transaction-level fraud data is confidential, so the dataset is simulated with Indian states, banks, rupee amounts and UPI transaction types. Fraud comes from known risk signals plus random noise. State and bank columns carry no fraud signal on purpose. Results show the METHOD, not real-world fraud performance. If you get permission to use real labelled data, keep the same column names and the notebook will run on it.
