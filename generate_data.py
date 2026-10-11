"""
Generates a SYNTHETIC Indian UPI transaction dataset for fraud/risk modelling.

WHY SYNTHETIC: real transaction-level data with fraud labels is confidential
(banks / NPCI do not publish it). This file simulates 100,000 UPI transactions
with Indian states, banks, rupee amounts and UPI transaction types.

Fraud is drawn from a probability score that depends on well-known risk signals
(new device, new payee, collect requests, night time, amount far above the
sender's normal, repeated failed attempts, new account ...) PLUS random noise,
so some frauds look normal and some normal transactions look risky -
like real life. State and bank columns carry NO fraud signal on purpose.

Run:  python generate_data.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 100_000

# ---------- helper to sample from a weighted dict ----------
def pick(options: dict, size):
    keys = list(options.keys())
    p = np.array(list(options.values()), dtype=float)
    return rng.choice(keys, size=size, p=p / p.sum())

# ---------- timestamps (2024), busier in the daytime/evening ----------
hour_w = np.array([1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 11, 10, 8, 5, 3], dtype=float)
hours = rng.choice(24, size=N, p=hour_w / hour_w.sum())
days = rng.integers(0, 366, size=N)  # 2024 is a leap year
ts = (pd.Timestamp("2024-01-01") + pd.to_timedelta(days, unit="D")
      + pd.to_timedelta(hours, unit="h")
      + pd.to_timedelta(rng.integers(0, 60, size=N), unit="m")
      + pd.to_timedelta(rng.integers(0, 60, size=N), unit="s"))

# ---------- transaction type and merchant category ----------
txn_type = pick({"P2P": .45, "P2M": .35, "Bill Payment": .10, "Recharge": .06, "Collect Request": .04}, N)
p2m_cat = {"Grocery": .25, "Food": .20, "Shopping": .18, "Fuel": .10, "Travel": .08,
           "Entertainment": .07, "Healthcare": .05, "Education": .04, "Online Gaming": .03}
category = np.empty(N, dtype=object)
category[txn_type == "P2P"] = "Peer Transfer"
category[txn_type == "Collect Request"] = "Peer Transfer"
category[txn_type == "Bill Payment"] = "Utilities"
category[txn_type == "Recharge"] = "Mobile Recharge"
m = txn_type == "P2M"
category[m] = pick(p2m_cat, m.sum())

# ---------- amount in INR (UPI limit is Rs 1 lakh) ----------
median_amt = {"Grocery": 450, "Food": 350, "Shopping": 1500, "Fuel": 1200, "Travel": 2500,
              "Entertainment": 600, "Healthcare": 1500, "Education": 8000, "Online Gaming": 800,
              "Peer Transfer": 1500, "Utilities": 1500, "Mobile Recharge": 300}
med = np.array([median_amt[c] for c in category], dtype=float)
amount = med * np.exp(rng.normal(0, 0.9, size=N))
# people often send round amounts to each other
is_peer = np.isin(txn_type, ["P2P", "Collect Request"])
round_mask = is_peer & (rng.random(N) < 0.6)
amount[round_mask] = np.round(amount[round_mask] / 50) * 50
amount = np.clip(amount, 1, 100_000).round(2)

# ---------- sender / device / behaviour columns ----------
states = {"Maharashtra": .13, "Uttar Pradesh": .12, "Karnataka": .08, "Tamil Nadu": .08, "Gujarat": .07,
          "Delhi": .07, "Rajasthan": .06, "West Bengal": .06, "Telangana": .06, "Madhya Pradesh": .05,
          "Andhra Pradesh": .05, "Kerala": .04, "Bihar": .04, "Haryana": .03, "Punjab": .03,
          "Odisha": .02, "Assam": .01}
banks = {"State Bank of India": .25, "HDFC Bank": .17, "ICICI Bank": .13, "Axis Bank": .10,
         "Punjab National Bank": .07, "Bank of Baroda": .07, "Kotak Mahindra Bank": .06,
         "Canara Bank": .05, "Union Bank": .04, "IndusInd Bank": .03, "Yes Bank": .03}
sender_state = pick(states, N)
sender_bank = pick(banks, N)
receiver_bank = pick(banks, N)
age_group = pick({"18-25": .22, "26-35": .33, "36-45": .22, "46-60": .15, "60+": .08}, N)
device = pick({"Android": .78, "iOS": .20, "Web": .02}, N)
network = pick({"4G": .45, "5G": .20, "WiFi": .30, "3G": .05}, N)

is_new_device = (rng.random(N) < 0.04).astype(int)
new_payee_p = np.select([txn_type == "Collect Request", txn_type == "P2P", txn_type == "P2M"], [0.80, 0.35, 0.15], 0.05)
is_new_payee = (rng.random(N) < new_payee_p).astype(int)
account_age = np.clip(np.exp(rng.normal(np.log(900), 0.9, size=N)), 1, 6000).astype(int)
txn_count_24h = rng.poisson(2.2, size=N)
failed_attempts = rng.poisson(0.15, size=N)
avg_amount_30d = (np.exp(rng.normal(np.log(900), 0.7, size=N))).round(2)

# ---------- fraud probability (logit) ----------
ratio = amount / avg_amount_30d
night = (hours <= 5).astype(int)
logit = (
    1.8 * is_new_device
    + 1.2 * is_new_payee
    + 2.2 * (txn_type == "Collect Request")
    + 1.2 * night
    + 1.0 * np.clip(np.log(ratio), 0, 4)
    + 0.8 * (amount > 20_000)
    + 0.9 * np.minimum(failed_attempts, 4)
    + 0.25 * np.maximum(txn_count_24h - 4, 0)
    + 1.2 * (account_age < 30)
    + 0.6 * (age_group == "60+")
    + 0.6 * (category == "Online Gaming")
    + 0.5 * ((category == "Peer Transfer") & (amount > 10_000))
    + 0.4 * (device == "Web")
    + 0.7 * (round_mask & (amount >= 5000))
    + rng.normal(0, 0.4, size=N)          # noise -> imperfect separation
)

# find the intercept that gives about 2% fraud overall
target_rate = 0.02
lo, hi = -15.0, 5.0
for _ in range(60):
    mid = (lo + hi) / 2
    rate = (1 / (1 + np.exp(-(logit + mid)))).mean()
    lo, hi = (mid, hi) if rate < target_rate else (lo, mid)
prob = 1 / (1 + np.exp(-(logit + mid)))
fraud = (rng.random(N) < prob).astype(int)

df = pd.DataFrame({
    "transaction_id": [f"UPI{2024}{i:07d}" for i in range(N)],
    "timestamp": ts,
    "transaction_type": txn_type,
    "merchant_category": category,
    "amount_inr": amount,
    "sender_state": sender_state,
    "sender_bank": sender_bank,
    "receiver_bank": receiver_bank,
    "sender_age_group": age_group,
    "device_type": device,
    "network_type": network,
    "is_new_device": is_new_device,
    "is_new_payee": is_new_payee,
    "sender_account_age_days": account_age,
    "txn_count_24h": txn_count_24h,
    "failed_attempts_24h": failed_attempts,
    "avg_txn_amount_30d": avg_amount_30d,
    "fraud_flag": fraud,
}).sort_values("timestamp").reset_index(drop=True)

# a little missing data, like real exports
for col, frac in [("network_type", 0.010), ("sender_state", 0.008), ("sender_account_age_days", 0.005)]:
    idx = rng.choice(N, size=int(N * frac), replace=False)
    df[col] = df[col].astype(object) if df[col].dtype != object else df[col]
    df.loc[idx, col] = np.nan

df.to_csv("upi_transactions_india.csv", index=False)
print("rows:", len(df), "| fraud rate: %.2f%%" % (df.fraud_flag.mean() * 100))
print(df.head())
