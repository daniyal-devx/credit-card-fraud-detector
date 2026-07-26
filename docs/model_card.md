# Model Card — Credit Card Fraud Detector

## What this model does

Predicts whether a credit card transaction is fraudulent, using a
Logistic Regression model trained on anonymized (PCA-transformed)
transaction data.

## Dataset

ULB Credit Card Fraud Detection dataset — 284,807 transactions over
two days, only 492 fraudulent (0.17%). Features `V1`-`V28` are PCA
components (not human-interpretable, anonymized for privacy). `Time`
and `Amount` are the only raw, meaningful columns.

## Why this is hard

At a 578:1 imbalance ratio, a model predicting "not fraud" for every
transaction would score 99.83% accuracy while catching zero fraud.
Accuracy is not a meaningful metric here — Precision-Recall AUC
(AUPRC) was used instead, as recommended by the dataset's own
documentation.

## Approaches compared

| Approach | Precision | Recall | AUPRC |
|---|---|---|---|
| Logistic Regression (`class_weight='balanced'`) | 0.06 | 0.92 | 0.717 |
| Logistic Regression + SMOTE | 0.06 | 0.92 | 0.725 |
| Isolation Forest (unsupervised anomaly detection) | 0.31 | 0.34 | — |

**SMOTE vs class_weight:** near-identical results. With only 394 real
fraud examples in training, SMOTE's synthetic points are interpolated
from the same limited signal `class_weight` already reweights for —
so the two methods converge on a similar outcome here.

**Isolation Forest:** a genuinely different tradeoff — far fewer false
alarms, but misses roughly two-thirds of real fraud, since it never
sees labeled fraud examples during training and instead flags
statistical outliers from "normal" behavior only.

## Chosen model

**Logistic Regression with `class_weight='balanced'`.** In this
context, a missed fraud case (false negative) carries a direct
financial loss, while a false alarm (false positive) costs a
recoverable customer-service interaction (e.g. verify identity,
unblock a card). Recall is prioritized accordingly.

## Threshold analysis

The default 0.5 probability threshold pushes recall to 0.92 but
collapses precision to 0.06. The Precision-Recall curve shows a
natural "knee" around recall ≈ 0.80, precision ≈ 0.83 — a
meaningfully better operating point for many real deployments,
trading a small amount of recall for far fewer false alarms. The
right threshold ultimately depends on the deployment context's
relative cost of a missed fraud case vs. a false alarm.

## Limitations

- PCA-anonymized features mean predictions are not individually
  interpretable in human terms (unlike the Churn Predictor project,
  SHAP explanations here would reference `V14`, `V17`, etc., which
  have no plain-language meaning).
- Trained on transactions from a two-day window in September 2013 —
  fraud patterns evolve over time; a production system would need
  periodic retraining.
- Train/test split was random-stratified, not time-based. A
  production fraud system should train only on past data to predict
  future transactions, to avoid unrealistic evaluation.
- Only 492 real fraud examples exist in total — any technique here is
  ultimately data-starved relative to the rarity of the event being
  predicted.

## Serving

Deployed via FastAPI (`/predict` endpoint). Input is scaled with a
`RobustScaler` (chosen over `StandardScaler` due to heavy-tailed
`Amount` outliers, e.g. a $25,691 transaction against a $22 median)
fit on training data only, then applied to incoming requests via
`.transform()` — never refit at inference time.
