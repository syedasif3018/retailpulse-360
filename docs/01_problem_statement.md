# 1. Problem Statement and Objectives

## 1.1 Context
An online retailer in the UK sells giftware, mostly to wholesale and repeat business customers. Keeping an existing customer usually costs less than finding a new one. If the retailer can see which active customers are likely to stop buying, the account or marketing team can contact them first, for example with an offer, a reminder or a call, before the revenue is lost.

The source data records transactions (invoice lines), not customers, and has no "churned" column. Part of this project is turning raw transactions into a labelled customer-level dataset without leaking future information.

## 1.2 Machine-learning task
**Supervised binary classification at customer level.**

> At an observation date (the *cutoff*), using only transactions on or before that date, predict whether an active customer will make **no valid purchase in the next 90 days**.

| Item | Definition |
|---|---|
| Unit of prediction | One customer at one cutoff date (a "customer snapshot") |
| Eligible customer | Has a known Customer ID and at least one valid purchase in the 365 days before the cutoff |
| Valid purchase | Non-cancelled invoice line with Quantity > 0 and Price > 0 |
| Target `churn_90d` | 1 = no valid purchase in (cutoff, cutoff + 90 days]; 0 = at least one valid purchase in that window |
| Features | Computed only from data in [cutoff − 365 days, cutoff], e.g. recency, frequency, monetary value, product variety, return behaviour, tenure, country |

**Why 90 days?** Most repeat customers in this business buy again within a few months. A window much shorter than the typical gap between purchases would label normal customers as churned. The audit and EDA check whether 90 days is reasonable by looking at the distribution of gaps between purchases.

## 1.3 Objectives and success criteria
1. **Data:** Audit, clean and document the transactions (duplicates, cancellations, missing customer IDs, invalid prices), with every decision written down.
2. **EDA:** Describe purchasing behaviour, and test at least one hypothesis statistically (for example, whether churners and retained customers differ in purchase frequency, using a Mann-Whitney U test).
3. **Modelling:** Build a naive baseline and at least three classifiers (Logistic Regression, Decision Tree, Random Forest, plus gradient boosting), tuned with a time-based validation split and preprocessed inside scikit-learn pipelines so the test data never influences preprocessing.
4. **Measurable success criteria**, all measured once on a held-out *later* time period:
   - ROC-AUC ≥ 0.75 and clearly above the baseline (0.50).
   - Churn-class recall ≥ 0.70 at the chosen threshold, so most at-risk customers are flagged, with precision reported alongside it.
   - PR-AUC above the churn base rate (the score a random model would get).
   - These targets are set before training. If the final model misses them, that is reported honestly.
5. **Interpretation:** Show which behaviours drive churn risk (permutation importance) and estimate the revenue held by high-risk customers.

## 1.4 Intended users and impact
- **Retention / CRM team:** a ranked list of at-risk customers to contact.
- **Sales management:** revenue at risk, broken down by segment and country.
- **Analysts:** a reusable, documented pipeline (scripts, notebooks, saved model, dashboard).

## 1.5 Assumptions and constraints
- Data covers Dec 2009 to Dec 2011 for a single retailer, so results are historical and not proof of how current retail customers behave.
- Rows without a Customer ID (guest or unknown buyers) cannot be followed over time and are excluded from customer-level modelling. They are kept for transaction-level EDA.
- "Churn" here is a business-defined, non-contractual proxy. A customer labelled churned may come back after the 90-day window.
- Time order must be respected: training cutoffs come before validation cutoffs, which come before test cutoffs, and features never use data after their cutoff.
- The 90-day label window means the last usable cutoff is about 90 days before the data ends (around 10 Sep 2011).
- Compute: a single laptop (CPU only). Cloud (Azure for Students) is used only to host the finished app, with cost safeguards.
