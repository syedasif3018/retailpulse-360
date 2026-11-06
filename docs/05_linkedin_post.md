# LinkedIn post (draft) and 60-second demo video plan

## Post text (about 1,700 characters; paste as-is, then attach the video)

Which customers are about to stop buying, before they actually do? 📉

I built **RetailPulse 360**, an end-to-end machine-learning project that predicts whether a business customer will make **no purchase in the next 90 days**.

🔹 **Data:** 1.07M real invoice lines from a UK online retailer (UCI Online Retail II, 2009–2011)
🔹 **Cleaning:** removed 34k duplicate rows (most because the two yearly sheets overlap on 1–9 Dec 2010) and orders that were cancelled within a day
🔹 **Target design:** 80.6% of repeat purchases happen within 90 days, so 90 days of silence is a meaningful churn signal
🔹 **No data leakage:** features use only the past, labels only the future, and models are tested on a later month they never saw
🔹 **Models:** two baselines vs Logistic Regression, Decision Tree, Random Forest and Gradient Boosting, tuned with scikit-learn pipelines

📊 **Result:** Gradient Boosting reached **ROC-AUC 0.77** on the unseen test month, clearly beating a simple "no order in 90 days" rule (0.71). The riskiest 10% of customers churned at **82%**.

💡 **Insight:** buying regularly every month matters far more than order size. Country had no link to churn (χ² p = 0.52).

🎓 **Key learning:** a random train/test split would have looked better and been wrong. Churn drops before Christmas, so only a time-based split gives honest numbers.

The model runs in a **Streamlit dashboard deployed on Azure** (free tier): risk bands, a priority contact list sorted by revenue at risk, a what-if scorer and batch scoring.

🔗 Code, notebooks and report: https://github.com/syedasif3018/retailpulse-360
🖥️ Live dashboard: https://retailpulse-360-541d.azurewebsites.net

I'd love feedback from people working in retail analytics or customer retention. How do you define churn when customers never cancel?

#MachineLearning #DataScience #Python #ScikitLearn #CustomerChurn #Streamlit #Azure #DataAnalytics

---

## 60-second screen-recording plan

**Before recording:**
- Open the live dashboard and wait until it has fully loaded, because the free tier can take about a minute to wake up.
- Use Chrome full screen (F11) at 100% zoom.
- Close other tabs and notifications.

**How to record (Windows 10):**
- Press **Win + Alt + R** to start and stop recording (Xbox Game Bar).
- The video is saved to Videos → Captures as an .mp4.
- Optionally trim it in the Photos app ("Edit & Create → Trim") or in Clipchamp.

| Time | Screen | What to show |
|---|---|---|
| 0–8 s | Overview tab | KPI row: customers, predicted churners, revenue at risk |
| 8–18 s | Overview | Hover over the risk-band bars and the risk-vs-value scatter |
| 18–28 s | Overview | Scroll to the priority contact list; move the "Show top" slider |
| 28–34 s | Sidebar | Filter Risk band = "Very high"; KPIs update |
| 34–44 s | Customer lookup | Pick a top customer; show their features vs a typical customer |
| 44–52 s | What-if | Lower "recency_days" or raise "active_months"; the probability drops |
| 52–60 s | Model performance | Metrics table and ROC / importance charts |

**Tips:**
- Move the mouse slowly and pause about 1 second on each result.
- No voice-over is needed; LinkedIn plays videos muted by default.
- Keep it at 60 seconds or less, landscape 1920×1080.
