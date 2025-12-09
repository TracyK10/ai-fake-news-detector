# Project Setup Guide

## Initial Setup

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

```bash
# Copy example environment file
copy .env.example .env  # Windows
# cp .env.example .env  # macOS/Linux

# Edit .env with your configuration
```

### 3. Download NLTK Data

The text preprocessor requires NLTK stopwords:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

## Training Your First Model

### Option 1: Using Sample Data (Quick Start)

```bash
# Generate sample dataset and train
cd ml/scripts
python data_loader.py
python train.py
```

### Option 2: Using Real Datasets (Recommended)

1. **Download Datasets**:
   - **WELFake** (Recommended): https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
   - **ISOT Fake News**: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
   - **LIAR**: https://github.com/thiagorainmaker77/liar_dataset

2. **Place in correct location**:
   ```
   ml/data/raw/WELFake_Dataset.csv          # Recommended
   ml/data/raw/Fake.csv                      # ISOT dataset
   ml/data/raw/True.csv                      # ISOT dataset
   ml/data/raw/liar_dataset.tsv              # LIAR dataset
   ```

3. **Process and train**:
   ```bash
   python ml/scripts/data_loader.py
   python ml/scripts/train.py
   ```

## Running the Application

### Backend

```bash
# From project root
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

### Using Docker

```bash
# Build and start all services
docker-compose up --build

# Access:
# - Frontend: http://localhost:3000
# - Backend: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

## Testing

### Backend Tests

```bash
pytest backend/tests/ -v --cov=backend
```

### Frontend Tests

```bash
cd frontend
npm test
```

## Troubleshooting

### Model Not Found Error

If you see "Model file not found" when starting the backend:

1. Ensure you've trained the model or have a pre-trained model
2. Check the `MODEL_PATH` in `.env` matches your model location
3. For development without a model, the `/health` endpoint will still work

### CUDA/GPU Issues

If you encounter CUDA errors:

1. Install CPU-only PyTorch:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. The code automatically falls back to CPU if CUDA is unavailable

### Frontend API Connection Issues

If the frontend can't connect to the backend:

1. Check backend is running on port 8000
2. Verify `REACT_APP_API_URL` in frontend `.env`
3. Check CORS settings in `backend/app/main.py`
