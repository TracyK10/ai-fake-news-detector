# Dataset Guide for AI Fake News Detector

This guide helps you obtain and prepare datasets for training the fake news detection model.

## ✅ Recommended Datasets (Working Links)

The old Kaggle competition dataset is no longer directly accessible. Use these alternatives instead:

### 1. WELFake Dataset ⭐ **RECOMMENDED**
- **Size**: 72,134 articles (35K real, 37K fake)
- **Source**: Merged from Kaggle, McIntire, Reuters, BuzzFeed
- **Download**: [Kaggle - WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- **File**: `WELFake_Dataset.csv` (245 MB)
- **Format**: CSV with columns: `Unnamed: 0`, `title`, `text`, `label` (0=fake, 1=real)

**Setup**:
```bash
# Download from Kaggle (requires Kaggle account)
# Then place in your project:
cp ~/Downloads/WELFake_Dataset.csv "ml/data/raw/"
```

### 2. ISOT Fake News Dataset
- **Size**: 44,919 articles (23.5K fake, 21.4K real)
- **Source**: ISOT Research Lab
- **Download**: [Kaggle - ISOT Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Files**: `Fake.csv` and `True.csv` (separate files)
- **Format**: CSV with columns: `title`, `text`, `subject`, `date`

**Setup**:
```bash
# Download both files and place in:
cp ~/Downloads/Fake.csv "ml/data/raw/"
cp ~/Downloads/True.csv "ml/data/raw/"
```

### 3. LIAR Dataset
- **Size**: 12,836 short statements
- **Source**: PolitiFact fact-checking
- **Download**: [GitHub - LIAR Dataset](https://github.com/thiagorainmaker77/liar_dataset)
- **Enhanced Version**: [LIAR-PLUS](https://github.com/Tariq60/LIAR-PLUS) (includes evidence)
- **Format**: TSV with 6-way labels (pants-fire, false, barely-true, half-true, mostly-true, true)

**Setup**:
```bash
git clone https://github.com/thiagorainmaker77/liar_dataset
cp liar_dataset/train.tsv "ml/data/raw/liar_dataset.tsv"
```

### 4. Alternative: Fake News Detection Dataset (2024)
- **Size**: 20,000 articles
- **Download**: [Kaggle - Fake News Detection](https://www.kaggle.com/datasets/mahdimashayekhi/fake-news-detection-dataset)
- **File**: `fake_news_dataset.csv` (34 MB)

## 🔧 Using the Datasets

### Quick Start with WELFake (Easiest)

1. Download WELFake dataset from Kaggle
2. Place `WELFake_Dataset.csv` in `ml/data/raw/`
3. Run the data loader (it will automatically detect the format):

```python
from ml.scripts.data_loader import DataLoader

loader = DataLoader()
df = loader.load_all_datasets()
loader.save_processed_data(df)
```

### Using ISOT Dataset

The ISOT dataset requires merging since fake and real news are in separate files. Update `data_loader.py`:

```python
def load_isot_dataset(self):
    fake_df = pd.read_csv('ml/data/raw/Fake.csv')
    fake_df['label'] = 1
    
    real_df = pd.read_csv('ml/data/raw/True.csv')
    real_df['label'] = 0
    
    # Combine title and text
    fake_df['text'] = fake_df['title'] + ' ' + fake_df['text']
    real_df['text'] = real_df['title'] + ' ' + real_df['text']
    
    combined = pd.concat([
        fake_df[['text', 'label']], 
        real_df[['text', 'label']]
    ])
    
    return combined
```

## 📊 Dataset Comparison

| Dataset | Size | Advantages | Disadvantages |
|---------|------|------------|---------------|
| **WELFake** | 72K | Large, diverse sources, ready to use | Mix of sources (varying quality) |
| **ISOT** | 44K | Clean, well-documented | Needs preprocessing |
| **LIAR** | 12.8K | High-quality fact-checks | Smaller, short statements only |

## 🚨 Important Notes

1. **Kaggle API Method** (Easiest):
   ```bash
   pip install kaggle
   # Get API key from kaggle.com/settings
   kaggle datasets download -d saurabhshahane/fake-news-classification
   unzip fake-news-classification.zip -d ml/data/raw/
   ```

2. **Manual Download**: All datasets can be downloaded manually from their Kaggle pages (requires free Kaggle account)

3. **Expected File Structure**:
   ```
   ml/data/raw/
   ├── WELFake_Dataset.csv       # WELFake (recommended)
   ├── Fake.csv                  # ISOT fake news
   ├── True.csv                  # ISOT real news
   └── liar_dataset.tsv          # LIAR dataset
   ```

## ✨ Next Steps

After downloading your dataset:

1. **Process the data**:
   ```bash
   python ml/scripts/data_loader.py
   ```

2. **Train the model**:
   ```bash
   python ml/scripts/train.py
   ```

3. **Monitor training**:
   ```bash
   tensorboard --logdir=runs
   ```

## 💡 Tips

- Start with **WELFake** - it's the largest and easiest to use
- For better accuracy, combine multiple datasets
- The sample dataset generator in `data_loader.py` can be used for testing without downloading
