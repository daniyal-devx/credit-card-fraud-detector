# Credit Card Fraud Detector

Detects fraudulent credit card transactions using Logistic Regression,
with a real-time FastAPI prediction endpoint, containerized with
Docker. Includes a full comparison of techniques for handling extreme
class imbalance.

## Why this matters

Fraud detection is a classic real-world ML problem where the event
you're trying to catch is extremely rare — missing one has a direct
financial cost, and the "obvious" metric (accuracy) is actively
misleading. This project focuses on choosing the right evaluation
approach and the right imbalance-handling technique for that
specific cost tradeoff, then serves the result as a real API instead
of a notebook.

## Dataset

[Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
— 284,807 transactions over two days, only 492 fraudulent (0.17%).
Features `V1`–`V28` are PCA-transformed components (anonymized,
not individually interpretable); `Time` and `Amount` are the only
raw, meaningful columns.

*Dataset not included in this repo (150MB) — download from the link
above and place at `data/creditcard.csv`.*

## Why this is harder than typical imbalanced classification

At a ~578:1 imbalance ratio, a model predicting "not fraud" for every
transaction scores 99.83% accuracy while catching zero fraud.
Accuracy is not a meaningful metric here — **Precision-Recall AUC
(AUPRC)** was used instead, per the dataset's own recommendation,
since it focuses entirely on how well the rare positive class is
handled.

## Approaches compared

| Approach | Precision | Recall | AUPRC |
|---|---|---|---|
| Logistic Regression (`class_weight='balanced'`) | 0.06 | 0.92 | 0.717 |
| Logistic Regression + SMOTE | 0.06 | 0.92 | 0.725 |
| Isolation Forest (unsupervised anomaly detection) | 0.31 | 0.34 | — |

**SMOTE vs. class_weight:** nearly identical results. With only 394
real fraud examples in training, SMOTE's synthetic points are
interpolated from the same limited signal `class_weight` already
reweights for — the two methods converge on a similar outcome.

**Isolation Forest:** a genuinely different tradeoff. Far fewer false
alarms (0.31 precision vs. 0.06), but misses roughly two-thirds of
real fraud, since it never sees labeled fraud examples during
training — it only learns what "normal" looks like and flags
statistical outliers.

## Chosen model

**Logistic Regression with `class_weight='balanced'`.** A missed
fraud case (false negative) carries a direct financial loss; a false
alarm (false positive) costs a recoverable customer-service
interaction. Recall is prioritized accordingly, and this was the
strongest recall among all three approaches.

## Threshold analysis

The default 0.5 probability threshold pushes recall to 0.92 but
collapses precision to 0.06. The Precision-Recall curve shows a
natural "knee" around recall ≈ 0.80, precision ≈ 0.83 — a
meaningfully better operating point for many deployments, trading a
small amount of recall for far fewer false alarms. See
`docs/pr_curve.png` and the full reasoning in `docs/model_card.md`.

## Preprocessing

- `V1`–`V28` are already PCA-scaled — no further scaling applied.
- `Time` and `Amount` scaled with **`RobustScaler`** (median/IQR),
  not `StandardScaler`, since `Amount` has heavy-tailed outliers
  (median $22 vs. a $25,691 maximum transaction) that would distort
  a mean/std-based scaler.
- Stratified 80/20 train/test split to preserve the true fraud ratio
  in both sets.

## Live Demo

Deployed on Render: **https://credit-card-fraud-detector-n932.onrender.com**

Interactive API docs: **https://credit-card-fraud-detector-n932.onrender.com/docs**

*Free tier note: the service sleeps after periods of inactivity — the
first request after idling may take 30-60 seconds to respond while it
wakes up.*

## API

Built with **FastAPI**, using **Pydantic** for automatic request
validation. Auto-generated interactive docs available at `/docs`.

### `POST /predict`

Accepts a transaction's `Time`, `Amount`, and `V1`–`V28`, returns:

```json
{
  "fraud_probability": 0.9421,
  "is_fraud": true
}
```

## Run locally

```bash
pip install -r requirements.txt
cd src
uvicorn app:app --reload
```

Visit `http://127.0.0.1:8000/docs` to test the API interactively.

## Run with Docker

```bash
cd src
docker build -t fraud-detector .
docker run -p 8000:8000 fraud-detector
```

## Project structure

```
credit-card-fraud-detector/
├── data/                    # raw dataset (gitignored — download separately)
├── notebooks/
│   └── 01_eda.ipynb         # EDA, cleaning, modeling, evaluation
├── src/
│   ├── app.py               # FastAPI app
│   ├── Dockerfile
│   ├── fraud_model.pkl      # trained Logistic Regression
│   └── scaler.pkl           # fitted RobustScaler
├── docs/
│   ├── model_card.md        # full model documentation
│   └── pr_curve.png         # precision-recall curve
├── requirements.txt
└── README.md
```

## Limitations

- PCA-anonymized features mean predictions aren't individually
  interpretable in human terms.
- Trained on a two-day window from September 2013 — fraud patterns
  evolve over time; a production system would need retraining.
- Train/test split was random-stratified, not time-based. A
  production system should train only on past data to predict
  future transactions.
- Only 492 real fraud examples exist in total — any technique here
  is fundamentally data-starved relative to the rarity of the event.

## What I'd improve

- Time-based train/test split instead of random-stratified
- Combine Isolation Forest as a first-pass filter with the supervised
  model as a second stage
- Add unit tests for the preprocessing pipeline
- Move off the free tier to avoid cold-start sleep delays

## Tech stack

Python · Pandas · Scikit-learn · imbalanced-learn (SMOTE) · FastAPI ·
Pydantic · Docker
