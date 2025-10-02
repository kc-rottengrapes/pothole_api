# 🌐 Deploy Pothole Detection API Globally

## Quick Deploy Options

### 1. **Render.com** (Recommended - Free Tier)
1. Push code to GitHub
2. Connect GitHub to Render.com
3. Create new Web Service
4. Use `requirements_cloud.txt`
5. Start command: `python app.py`

### 2. **Railway.app** (Easy Deploy)
1. Push to GitHub
2. Connect to Railway
3. Auto-deploys with `railway.json`

### 3. **Heroku** (Classic Option)
```bash
heroku create your-pothole-api
git push heroku main
```

### 4. **Google Cloud Run** (Scalable)
```bash
gcloud run deploy --source .
```

## Files for Deployment:
- `app.py` - Main API (cloud-ready)
- `requirements_cloud.txt` - Dependencies
- `best.pt` - Your trained model
- `Dockerfile` - Container config
- `render.yaml` - Render config
- `railway.json` - Railway config

## Test Locally:
```bash
pip install -r requirements_cloud.txt
python app.py
```
Visit: http://localhost:8000

## After Deployment:
Your API will be accessible globally at:
- `https://your-app.render.com`
- `https://your-app.railway.app`
- etc.

Users worldwide can upload images and get instant pothole detection!