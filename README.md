# 🔍 AI Fake News Detector

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18.2-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)
[![Build](https://img.shields.io/badge/Build-Passing-success?style=for-the-badge)](https://github.com/TracyK10/ai-fake-news-detector/actions)

> A production-grade full-stack application for detecting misinformation using fine-tuned Transformer models (BERT/RoBERTa). Features a real-time feedback loop where user corrections continuously improve model accuracy.

---

## 📊 Project Overview

This application leverages **state-of-the-art NLP** to analyze news articles and classify them as authentic or fabricated. Built end-to-end with modern software engineering practices, it demonstrates competency across machine learning, backend API development, frontend UI/UX, and DevOps.

**Key Features:**
- **Deep Learning Pipeline**: Fine-tuned RoBERTa model on 50K+ labeled news articles
- **RESTful API**: Async FastAPI backend with rate limiting and input validation
- **Interactive Frontend**: React SPA with real-time confidence visualization
- **Feedback Loop**: User corrections stored for continuous model improvement
- **Containerized Deployment**: Docker-based architecture with CI/CD automation

---

## 🏗️ System Architecture

The application follows a **microservices architecture** with clear separation of concerns:

```
┌─────────────┐      HTTP/JSON      ┌──────────────┐      Inference      ┌─────────────┐
│   React     │ ───────────────────▶ │   FastAPI    │ ──────────────────▶ │  RoBERTa    │
│   Frontend  │ ◀─────────────────── │   Backend    │ ◀────────────────── │   Model     │
└─────────────┘   Predictions (%)    └──────────────┘   Logits/Probs     └─────────────┘
                                              │
                                              ▼
                                      ┌──────────────┐
                                      │   SQLite     │
                                      │  Feedback DB │
                                      └──────────────┘
```

**Data Flow:**
1. User submits text via React interface
2. Frontend sends POST request to `/api/v1/analyze`
3. FastAPI validates input (Pydantic schemas)
4. Text is tokenized and passed to the Transformer model
5. Model returns prediction probabilities
6. Response includes label (`Real`/`Fake`) and confidence score
7. User feedback stored in database for retraining

**Infrastructure:**
- **Containerization**: Multi-stage Docker builds for optimized images
- **Orchestration**: Docker Compose manages service dependencies
- **CI/CD**: GitHub Actions runs tests, linting, and security scans on every push

![Architecture Diagram](docs/architecture.png)

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|--------------|
| **ML/AI** | PyTorch, Hugging Face Transformers, Scikit-learn, Pandas, NLTK |
| **Backend** | FastAPI, Uvicorn (ASGI), SQLAlchemy, Pydantic, SlowAPI |
| **Frontend** | React 18, Vite, Tailwind CSS, Axios |
| **Database** | SQLite (dev), PostgreSQL-ready (prod) |
| **DevOps** | Docker, Docker Compose, GitHub Actions, Trivy (security) |
| **Testing** | Pytest, React Testing Library |

---

## 📈 Model Performance

The model was fine-tuned on a merged dataset from **LIAR**, **Kaggle Fake News**, and **FakeNewsNet** (total: ~50K samples).

| Metric | Score |
|--------|-------|
| **Accuracy** | 92.7% |
| **F1-Score** | 0.92 |
| **Precision** | 91.4% |
| **Recall** | 92.9% |

**Training Configuration:**
- Base Model: `roberta-base` (125M parameters)
- Optimizer: AdamW with linear warmup
- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 3
- Max Sequence Length: 512 tokens

![Training Loss](docs/training_loss.png)

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (recommended)

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/TracyK10/ai-fake-news-detector.git
cd ai-fake-news-detector

# Start all services
docker-compose up --build

# Access:
# - Frontend: http://localhost:3000
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### Option 2: Manual Setup

#### Backend
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"

# Train model (or download pre-trained)
python ml/scripts/data_loader.py
python ml/scripts/train.py

# Start API server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend
```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Access the application at `http://localhost:3000`

---

## 📡 API Documentation

### `POST /api/v1/analyze`
Analyzes news text and returns authenticity prediction.

**Request:**
```json
{
  "text": "Breaking: Scientists discover new planet capable of sustaining human life..."
}
```

**Response:**
```json
{
  "label": "Fake",
  "confidence_score": 0.87,
  "probabilities": {
    "real": 0.13,
    "fake": 0.87
  }
}
```

**Rate Limit:** 10 requests/minute per IP

---

### `POST /api/v1/feedback`
Submits user correction for model improvement.

**Request:**
```json
{
  "text": "Original news article text...",
  "predicted_label": "Fake",
  "confidence_score": 0.87,
  "user_correction": "Real"
}
```

**Response:**
```json
{
  "message": "Thank you for your feedback!",
  "feedback_id": 42
}
```

---

### `GET /health`
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true
}
```

---

## 🖼️ Screenshots

### Dashboard - Analysis Interface
![Dashboard View](docs/dashboard.png)

### Model Training Metrics
![Training Graphs](docs/training_loss.png)

---

## 🧪 Testing

```bash
# Backend tests with coverage
pytest backend/tests/ -v --cov=backend --cov-report=html

# Frontend tests
cd frontend && npm test

# Linting
flake8 backend/ ml/
```

**CI/CD Pipeline** automatically runs:
- Unit & integration tests
- Code linting (Flake8)
- Security scanning (Trivy)
- Docker build verification

---

## 📁 Project Structure

```
ai-fake-news-detector/
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── api/routes/     # Endpoint definitions
│   │   ├── core/           # Config & schemas
│   │   ├── database/       # SQLAlchemy models
│   │   └── services/       # Business logic
│   └── tests/              # Pytest suite
├── frontend/               # React application
│   └── src/
│       ├── components/     # UI components
│       ├── services/       # API client
│       └── styles/         # Tailwind CSS
├── ml/                     # ML pipeline
│   ├── data/              # Datasets
│   ├── models/            # Trained weights
│   └── scripts/           # Training & inference
├── .github/workflows/      # CI/CD automation
└── docker-compose.yml      # Service orchestration
```

---

## 🔐 Security Features

- **Input Validation**: Pydantic schemas prevent injection attacks
- **Rate Limiting**: SlowAPI prevents abuse (10 req/min)
- **CORS**: Restricted to allowed origins
- **Dependency Scanning**: Trivy checks for CVEs in CI/CD
- **Environment Variables**: Secrets managed via `.env`

---

## 🚢 Deployment

### Backend
- **Cloud Run** (Google Cloud): Serverless autoscaling
- **EC2** (AWS): Traditional VM deployment with Nginx
- **Heroku**: Container-based deployment

### Frontend
- **Vercel**: Optimized for React/Vite (recommended)
- **Netlify**: Static site hosting with CDN
- **S3 + CloudFront**: AWS-native solution

---

## 🎯 Future Enhancements

- [ ] **Active Learning**: Prioritize uncertain predictions for human review
- [ ] **Multi-language Support**: Extend to Spanish, French, German
- [ ] **Explainability**: SHAP values to highlight influential words
- [ ] **URL Scraping**: Analyze articles directly from URLs
- [ ] **Ensemble Models**: Combine BERT + RoBERTa for higher accuracy

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Tracy Karanja**  
[GitHub](https://github.com/TracyK10) • [LinkedIn](https://www.linkedin.com/in/tracy-karanja/) • [Email](mailto:tkaranja@andrew.cmu.edu)

---

<div align="center">
  <sub>Built with ❤️ for Summer 2026 Internships</sub>
</div>
