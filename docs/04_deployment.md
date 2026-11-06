# 4. Cloud Deployment (Azure App Service, Free tier)

**Live dashboard:** https://retailpulse-360-541d.azurewebsites.net

The free tier sleeps when idle, so the first visit after a quiet period can take about a minute to load.

## Architecture
```
GitHub repo ──(deploy/make_package.py)──► retailpulse-app.zip (0.74 MB)
                                              │  az webapp deploy
                                              ▼
Azure App Service (Linux, Python 3.14, Free F1, East Asia)
  └─ Oryx build: pip install -r requirements.txt (requirements-app.txt)
  └─ startup: python -m streamlit run app/streamlit_app.py --server.port 8000
       ├─ models/churn_model.joblib      (trained Gradient Boosting pipeline)
       ├─ app/data/scores_*.csv          (pre-scored customer snapshots)
       └─ reports/ (metrics JSON + 4 figures)
```
Training happens locally (notebook 03). Only the finished model and pre-computed scores are deployed. The raw data and training code are not needed in the cloud.

## Resources and cost
| Resource | Name | Tier | Cost |
|---|---|---|---|
| Resource group | `rg-retailpulse` (East Asia) | n/a | free |
| App Service plan | `asp-retailpulse` | Free F1 (Linux) | **$0** |
| Web app | `retailpulse-360-541d` | runs on the plan above | **$0** |

- **Subscription:** Azure for Students. No credit card is attached, so usage cannot create a bill, and the F1 tier does not consume the $100 credit.
- **Region:** the student policy only allows East Asia, Malaysia West, Poland Central, Korea Central and Indonesia Central. East Asia was chosen as the nearest to India.
- **F1 limits:** 60 CPU-minutes/day, 1 GB RAM, no "Always On", so the app sleeps when idle.

## How it was deployed (Windows PowerShell)
```powershell
az login                                                    # browser sign-in, choose "Azure for Students"
az provider register -n Microsoft.Web --wait                # one-time: enable App Service
az group create -n rg-retailpulse -l eastasia
az appservice plan create -g rg-retailpulse -n asp-retailpulse --is-linux --sku F1 -l eastasia
az webapp create -g rg-retailpulse -p asp-retailpulse -n retailpulse-360-541d --runtime "PYTHON:3.14" --https-only true
az webapp config set -g rg-retailpulse -n retailpulse-360-541d --web-sockets-enabled true `
  --startup-file "python -m streamlit run app/streamlit_app.py --server.port 8000 --server.address 0.0.0.0"
az webapp config appsettings set -g rg-retailpulse -n retailpulse-360-541d `
  --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true WEBSITES_PORT=8000
python deploy/make_package.py                               # -> dist/retailpulse-app.zip
az webapp deploy -g rg-retailpulse -n retailpulse-360-541d --src-path dist/retailpulse-app.zip --type zip
```
**Why each setting matters:**
- **WebSockets on:** Streamlit's browser-server connection uses a WebSocket.
- **Port 8000:** the port App Service forwards traffic to.
- **Build during deployment:** Azure installs the Python packages from `requirements.txt`.
- **HTTPS only:** plain HTTP is redirected to HTTPS.

## Verification (7 Nov 2026)
- `GET /_stcore/health` returns `ok`.
- The home page returns HTTP 200; `http://` redirects (301) to `https://`.
- The WebSocket handshake on `/_stcore/stream` returns HTTP 101 (protocol switch accepted).

## Update or remove
```powershell
python deploy/make_package.py; az webapp deploy -g rg-retailpulse -n retailpulse-360-541d --src-path dist/retailpulse-app.zip --type zip
az group delete -n rg-retailpulse --yes        # deletes everything above
```
