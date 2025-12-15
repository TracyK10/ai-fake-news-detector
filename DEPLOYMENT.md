# Deployment Guide - AI Fake News Detector

Complete guide for deploying your full-stack application to production.

---

## 🎯 Deployment Architecture

**Frontend**: Vercel (Free tier)  
**Backend**: Render (Free tier)  
**Model Storage**: Hugging Face Hub or GitHub LFS  
**Database**: SQLite (included) or upgrade to PostgreSQL on Render

---

## 🌐 Part 1: Deploy Frontend to Vercel

### Step 1: Prepare Frontend for Deployment

1. **Update API endpoint** to use environment variable:

Create `frontend/.env.production`:
```env
VITE_API_URL=https://your-backend-url.onrender.com
```

2. **Update `frontend/src/services/api.js`**:
```javascript
import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const analyzeNews = async (text) => {
  const response = await api.post('/api/v1/analyze', { text });
  return response.data;
};

export const submitFeedback = async (feedbackData) => {
  const response = await api.post('/api/v1/feedback', feedbackData);
  return response.data;
};
```

### Step 2: Deploy to Vercel

**Option A: Vercel CLI (Recommended)**
```bash
# Install Vercel CLI
npm install -g vercel

# Navigate to frontend directory
cd frontend

# Login to Vercel
vercel login

# Deploy
vercel

# Follow prompts:
# - Set up and deploy? Yes
# - Which scope? Your account
# - Link to existing project? No
# - Project name? ai-fake-news-detector
# - Directory? ./
# - Override settings? No
```

**Option B: Vercel Dashboard (Easier for beginners)**
1. Go to [vercel.com](https://vercel.com)
2. Sign up with GitHub
3. Click "Add New Project"
4. Import your `ai-fake-news-detector` repository
5. Configure:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`
6. Add Environment Variable:
   - Key: `VITE_API_URL`
   - Value: (Leave empty for now, update after backend deployment)
7. Click "Deploy"

### Step 3: Configure Custom Domain (Optional)
- Go to Project Settings → Domains
- Add custom domain or use Vercel's `*.vercel.app` domain

---

## 🖥️ Part 2: Deploy Backend to Render

### Step 1: Prepare Backend for Deployment

1. **Create `render.yaml`** in project root:
```yaml
services:
  - type: web
    name: ai-fake-news-backend
    env: python
    buildCommand: "pip install -r requirements.txt && python -c 'import nltk; nltk.download(\"stopwords\")'"
    startCommand: "uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.12
      - key: MODEL_PATH
        value: ml/models/best_model.pt
      - key: FRONTEND_URL
        value: https://your-frontend.vercel.app
```

2. **Update CORS in `backend/app/main.py`**:
```python
from backend.app.core.config import get_settings
import os

settings = get_settings()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL,
        os.getenv("FRONTEND_URL", "http://localhost:3000"),
        "https://*.vercel.app",  # Allow all Vercel deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

3. **Update `backend/app/core/config.py`**:
```python
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    APP_NAME: str = "AI Fake News Detector API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = int(os.getenv("PORT", 8000))
    
    # Frontend URL (for CORS)
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "ml/models/best_model.pt")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./feedback.db")
    
    class Config:
        env_file = ".env"
```

### Step 2: Handle Model File (CRITICAL!)

Your model file is **500MB+**. GitHub has a 100MB file limit. Options:

**Option A: Hugging Face Hub (Recommended)**
```bash
# Install huggingface_hub
pip install huggingface_hub

# Upload model
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='ml/models/best_model.pt',
    path_in_repo='best_model.pt',
    repo_id='TracyK10/fake-news-detector',
    repo_type='model',
)
"
```

Then update `inference.py` to download on startup:
```python
from huggingface_hub import hf_hub_download
import os

def __init__(self, model_path: str = "ml/models/best_model.pt"):
    self.model_path = Path(model_path)
    
    # Download from HuggingFace if not present
    if not self.model_path.exists():
        logger.info("Downloading model from HuggingFace Hub...")
        model_file = hf_hub_download(
            repo_id="TracyK10/fake-news-detector",
            filename="best_model.pt",
            cache_dir="ml/models"
        )
        self.model_path = Path(model_file)
```

**Option B: GitHub LFS**
```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.pt"
git add .gitattributes

# Commit and push
git add ml/models/best_model.pt
git commit -m "Add model with LFS"
git push
```

**Option C: Exclude model (use smaller version)**
For demo purposes, you could train on a smaller subset or use a lighter model.

### Step 3: Deploy to Render

1. **Sign up at [render.com](https://render.com)** with GitHub

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your `ai-fake-news-detector` repository
   - Configure:
     - **Name**: `ai-fake-news-backend`
     - **Region**: Choose closest to you
     - **Branch**: `main`
     - **Root Directory**: (leave empty)
     - **Runtime**: Python 3
     - **Build Command**: 
       ```bash
       pip install -r requirements.txt && python -c "import nltk; nltk.download('stopwords')"
       ```
     - **Start Command**: 
       ```bash
       uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
       ```

3. **Add Environment Variables**:
   - `PYTHON_VERSION` = `3.10.12`
   - `MODEL_PATH` = `ml/models/best_model.pt`
   - `FRONTEND_URL` = `https://your-app.vercel.app` (update after Vercel deployment)

4. **Create Service** (Free tier)

5. **Wait for deployment** (~5-10 minutes first time)

### Step 4: Get Backend URL
- Copy your backend URL: `https://ai-fake-news-backend.onrender.com`

---

## 🔗 Part 3: Connect Frontend to Backend

1. **Update Vercel Environment Variable**:
   - Go to Vercel Dashboard → Your Project → Settings → Environment Variables
   - Update `VITE_API_URL` to your Render backend URL
   - Redeploy: `vercel --prod`

2. **Update Render Environment Variable**:
   - Go to Render Dashboard → Your Service → Environment
   - Update `FRONTEND_URL` to your Vercel URL
   - Render will auto-redeploy

3. **Test the connection**:
   - Visit your Vercel URL
   - Try analyzing sample text
   - Check browser console for errors

---

## ⚡ Alternative Backend Hosting Options

### **Railway.app** (Good alternative to Render)
- Free tier: $5 credit/month
- GitHub integration
- Easy setup
- [railway.app](https://railway.app)

### **Fly.io** (Best for Docker)
- Free tier: 3 VMs
- Great Docker support
- Global deployment
- [fly.io](https://fly.io)

### **Google Cloud Run** (Serverless)
- Pay-per-use
- Auto-scaling
- Free tier: 2 million requests/month
- Requires credit card

### **Python Anywhere** (Python-specific)
- Free tier available
- Web-based IDE
- Simple deployment
- [pythonanywhere.com](https://pythonanywhere.com)

---

## 🚨 Common Deployment Issues & Fixes

### Issue 1: CORS Errors
**Solution**: Make sure `FRONTEND_URL` environment variable is set correctly in backend

### Issue 2: Model Loading Fails
**Solutions**:
- Check model file size (Render has 512MB limit on free tier)
- Use Hugging Face Hub to download model on startup
- Consider using a smaller model variant

### Issue 3: Build Timeout
**Solution**: 
- Reduce dependencies in `requirements.txt`
- Use lighter PyTorch version: `torch --index-url https://download.pytorch.org/whl/cpu`

### Issue 4: Port Binding Error
**Solution**: Use `$PORT` environment variable in start command (Render provides this)

### Issue 5: Database Issues
**Solution**: SQLite works on Render, but consider upgrading to PostgreSQL for production:
```bash
# Render provides PostgreSQL addon (free tier: 1GB)
# Update DATABASE_URL in environment variables
```

---

## 📊 Monitoring & Debugging

### Vercel
- **Logs**: Dashboard → Deployments → Click deployment → Logs
- **Analytics**: Dashboard → Analytics
- **Error tracking**: Automatic

### Render
- **Logs**: Dashboard → Service → Logs tab
- **Metrics**: Dashboard → Service → Metrics
- **Shell access**: Dashboard → Service → Shell

---

## 🎓 Portfolio Tips

1. **Add deployed links to README**:
   ```markdown
   ## 🌐 Live Demo
   - **Frontend**: https://your-app.vercel.app
   - **API Docs**: https://your-backend.onrender.com/docs
   ```

2. **Create demo video** showing the deployed app

3. **Add to LinkedIn/Resume**:
   - "Deployed full-stack ML application to Vercel (frontend) and Render (backend)"
   - "Integrated Hugging Face Hub for model storage"

4. **Monitor uptime**:
   - Render free tier sleeps after 15 min inactivity
   - Consider using [UptimeRobot](https://uptimerobot.com) to ping it

---

## ✅ Deployment Checklist

Before going live:
- [ ] Test locally with production build (`npm run build` + `npm run preview`)
- [ ] Update all environment variables
- [ ] Test API endpoints (`/health`, `/docs`)
- [ ] Check CORS configuration
- [ ] Verify model loading
- [ ] Test with sample data
- [ ] Monitor initial deployment logs
- [ ] Update README with live links
- [ ] Test on mobile devices
- [ ] Share with friends for testing!

---

## 🚀 Quick Deploy Commands

```bash
# Frontend (Vercel)
cd frontend
vercel --prod

# Backend (after Render is set up)
git add .
git commit -m "Deploy to production"
git push origin main  # Render auto-deploys
```

---

Good luck with your deployment! 🎉
