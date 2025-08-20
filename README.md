# 🎓 Scholar Scraper

```
  ____       _           _             ____                                   
 / ___|  ___| |__   ___ | | __ _ _ __  / ___|  ___ _ __ __ _ _ __   ___ _ __     
 \___ \ / __| '_ \ / _ \| |/ _` | '__| \___ \ / __| '__/ _` | '_ \ / _ \ '__|    
  ___) | (__| | | | (_) | | (_| | |     ___) | (__| | | (_| | |_) |  __/ |       
 |____/ \___|_| |_|\___/|_|\__,_|_|    |____/ \___|_|  \__,_| .__/ \___|_|       
                                                            |_|                  
```

🚀 **Scholar Scraper** is a comprehensive, production-ready Python toolkit for predicting who will participate in top AI/ML conferences (AAAI, NeurIPS, ICLR). It combines intelligent web scraping, advanced ML modeling, and intuitive data exploration to deliver actionable insights with 90.7% accuracy in matching predictions to real scholar profiles.

**✨ What makes this special:** Unlike simple citation scrapers, this system uses temporal feature engineering and ensemble modeling to predict *future* participation, not just analyze past patterns. The conservative thresholding approach ensures high precision (68% precision, 77% AUC) suitable for real-world decision making.

---

## 🚀 Quick Start (5 Minutes)

**Want to get started immediately?** Follow these steps:

```bash
# 1. Clone and setup
git clone https://github.com/krishnanefx/Scholar_Scraper.git
cd Scholar_Scraper
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Run a prediction model (uses existing data)
python src/models/aaai_predict_authors.py --quiet-warnings

# 3. Generate comprehensive Excel report with matched profiles
python src/conference_predictions_summary_2026.py

# 4. Explore results interactively
streamlit run src/dashboard.py
```

**🎯 What you get:** A machine learning system that predicts who will attend AAAI 2026, NeurIPS 2025, and ICLR 2026 with 90.7% profile matching accuracy. The system analyzes 42K+ authors and outputs Excel reports with complete scholar profiles, institutional affiliations, and confidence scores.

---

## ✨ Core Features & Architecture

### 🧠 Machine Learning Pipeline
- **🎯 Temporal Feature Engineering**: 15+ features including participation streaks, exponential decay scoring, Markov transitions, and recency weighting
- **🤖 Ensemble Modeling**: VotingClassifier combining Gradient Boosting + Logistic Regression with isotonic calibration  
- **📊 Conservative Thresholding**: 85th percentile selection to achieve 68% precision (fewer false positives)
- **🔄 GroupKFold Cross-Validation**: Prevents data leakage by ensuring same author doesn't appear in train/test splits

### 🕷️ Intelligent Data Collection  
- **📚 Multi-Source Scraping**: Google Scholar + Wikipedia with automatic fallback from requests to Selenium when blocked
- **� Smart Caching**: Persistent joblib/JSON caches for Wikipedia lookups and fuzzy matching (10x speedup on repeated runs)
- **🔍 Profile Enrichment**: Institution mapping, research interest classification, citation metrics
- **📋 Queue Management**: Resumable BFS crawling with `queue.txt` and incremental processing

### 🔗 Data Integration & Matching
- **🎯 Fuzzy Name Matching**: 80% similarity threshold with fuzzywuzzy for linking predictions to profiles
- **📈 Profile Completeness**: 90.7% match rate between predictions and scholar database (4,960/5,467 authors)
- **📊 Multi-Conference Support**: AAAI, NeurIPS, ICLR with unified prediction pipeline
- **� Production Outputs**: Excel reports with separate sheets per conference + summary statistics

---

## 🏗️ System Architecture & Design Decisions

### Why This Architecture Works

**1. 🎯 Temporal Features Over Static Features**
```python
# Instead of just "total publications", we compute:
features = {
    'num_participations': len(past_years),
    'max_consecutive_years': calculate_streaks(years),
    'exp_decay_sum': sum(exp(-0.5 * (current_year - y)) for y in years),
    'markov_prob': consecutive_transitions / total_transitions,
    'years_since_last': current_year - max(years)
}
```
**Why:** Academic careers are temporal - recent activity predicts future participation better than career totals.

**2. 🤖 Ensemble + Calibration Instead of Single Model**  
```python
ensemble = VotingClassifier([
    ('gb', GradientBoostingClassifier),  # Captures feature interactions
    ('lr', LogisticRegression)           # Provides stable linear signal
])
calibrated_model = CalibratedClassifierCV(ensemble, method='isotonic')
```
**Why:** Gradient boosting handles complex patterns while logistic regression prevents overfitting. Calibration ensures probabilities are well-calibrated for threshold selection.

**3. 📊 Conservative Thresholding (85th percentile)**
```python
conservative_threshold = np.percentile(training_probabilities, 85)
predictions = probabilities >= conservative_threshold
```
**Why:** For downstream human review, precision matters more than recall. Better to predict 1,669 highly likely participants than 5,000 uncertain ones.

**4. 🔄 GroupKFold Cross-Validation**
```python
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=author_names):
    # Same author never appears in both train and test
```
**Why:** Prevents data leakage - same author's historical patterns shouldn't appear in both training and validation.

---

## � Repository Structure (What Each File Does)

```
Scholar_Scraper/
├── � data/
│   ├── raw/                                    # Input datasets
│   │   ├── aaai25_papers_authors_split.csv     # AAAI 2010-2025 publications
│   │   ├── iclr_2020_2025.parquet             # ICLR historical data  
│   │   └── neurips_2020_2024_combined_data.parquet # NeurIPS historical data
│   ├── processed/                              # Generated models & profiles
│   │   ├── scholar_profiles.csv               # 42K+ crawled profiles
│   │   ├── aaai_participation_model.pkl       # Trained AAAI model
│   │   ├── neurips_participation_model.pkl    # Trained NeurIPS model
│   │   └── iclr_participation_model.pkl       # Trained ICLR model
│   └── predictions/                            # Model outputs
│       ├── aaai_2026_predictions.csv          # 1,669 AAAI predictions
│       ├── neurips_2025_predictions.csv       # 2,194 NeurIPS predictions
│       └── iclr_2026_predictions.csv          # 1,604 ICLR predictions
├── 🔧 src/
│   ├── scrapers/                               # Data collection
│   │   ├── main.py                            # 🎯 BFS Google Scholar crawler
│   │   ├── aaai_scraper.py                    # AAAI conference scraper
│   │   ├── neurips_scraper.py                 # NeurIPS API scraper
│   │   └── iclr_scraper.py                    # ICLR API scraper  
│   ├── models/                                 # ML prediction pipelines
│   │   ├── aaai_predict_authors.py            # 🎯 AAAI 2026 predictions
│   │   ├── neurips_predict_authors.py         # NeurIPS 2025 predictions
│   │   └── iclr_predict_authors.py            # ICLR 2026 predictions
│   ├── conference_predictions_summary_2026.py # 🎯 Match predictions with profiles
│   └── dashboard.py                           # 🎯 Streamlit exploration interface
├── � cache/                                   # Performance optimization
│   ├── wiki_lookup_cache.joblib              # Wikipedia API cache
│   ├── fuzzy_match_cache.joblib              # Name matching cache
│   └── queue.txt                              # Crawler queue state
├── 📈 outputs/
│   └── Conference_Predictions_with_Scholar_Profiles.xlsx # 🎯 Final report
└── 🧪 tests/                                   # Unit tests
    └── test_path_resolution.py               # Path resolution tests
```

**🎯 Key files to run:**
- `src/scrapers/main.py` - Builds the scholar database  
- `src/models/aaai_predict_authors.py` - Generates predictions
- `src/conference_predictions_summary_2026.py` - Creates final Excel report
- `src/dashboard.py` - Interactive exploration

---

## 🎯 Step-by-Step Usage Guide

### Option 1: � Quick Prediction Run (5 minutes)
Use existing data to generate conference predictions immediately:

```bash
# Setup environment
git clone https://github.com/krishnanefx/Scholar_Scraper.git
cd Scholar_Scraper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate predictions using existing historical data
python src/models/aaai_predict_authors.py --quiet-warnings
python src/models/neurips_predict_authors.py --quiet-warnings  
python src/models/iclr_predict_authors.py --quiet-warnings

# Create comprehensive Excel report with scholar profiles
python src/conference_predictions_summary_2026.py --year 2026

# Launch interactive dashboard
streamlit run src/dashboard.py
```

**📊 Output:** Excel file at `outputs/Conference_Predictions_with_Scholar_Profiles.xlsx` with 5,467 predictions across all conferences, 90.7% matched to full scholar profiles.

### Option 2: 🕷️ Full Pipeline with Fresh Data Collection

```bash
# 1. Collect fresh scholar profiles (takes 2-8 hours depending on target size)
python src/scrapers/main.py
# 🔄 This runs BFS crawling starting from Yoshua Bengio's profile
# 💾 Creates data/processed/scholar_profiles.csv with enriched profiles
# 📋 Use src/dashboard.py to monitor progress and add new seeds

# 2. Optional: Scrape fresh conference data  
python src/scrapers/aaai_scraper.py      # Updates AAAI papers
python src/scrapers/neurips_scraper.py   # Updates NeurIPS data
python src/scrapers/iclr_scraper.py      # Updates ICLR data

# 3. Train models and generate predictions
python src/models/aaai_predict_authors.py
python src/models/neurips_predict_authors.py
python src/models/iclr_predict_authors.py

# 4. Create final report
python src/conference_predictions_summary_2026.py
```

### Option 3: 🔧 Custom Conference/Year Predictions

```bash
# Predict for specific year
python src/models/aaai_predict_authors.py --data-path data/raw/custom_aaai.csv

# Use environment variables for reproducible runs
export AAAI_DATA_PATH=data/raw/aaai25_papers_authors_split.csv
export AAAI_OUTPUT_DIR=experiments/run_001/predictions
export AAAI_MODEL_PATH=experiments/run_001/models/aaai_model.pkl
python src/models/aaai_predict_authors.py --quiet-warnings

# Generate report for specific year and threshold
python src/conference_predictions_summary_2026.py --year 2027 --threshold 75
```

### 🎯 Understanding the Outputs

**1. 📊 Individual Prediction CSVs** (`data/predictions/`)
```csv
predicted_author,will_participate_2026,participation_probability,confidence_percent,rank
Peter Stone,1,1.000,100.0,1
Yu Cheng,1,0.987,98.7,2
```

**2. 📈 Excel Report** (`outputs/Conference_Predictions_with_Scholar_Profiles.xlsx`)
- **AAAI_2026 sheet**: 1,528 matched participants with full academic profiles
- **NeurIPS_2025 sheet**: 1,983 matched participants  
- **ICLR_2026 sheet**: 1,449 matched participants
- **Summary sheet**: Cross-conference statistics and match rates
- **Unmatched_Authors sheet**: 507 entries for manual review

**3. 🎨 Interactive Dashboard** (`streamlit run src/dashboard.py`)
- Search and filter 42K+ scholar profiles
- View conference participation patterns
- Export filtered results as CSV
- Network visualization tools

---

## 📊 Performance Results & Model Analysis

### 🎯 Real Performance Metrics (From Live Run)

**AAAI 2026 Prediction Results:**
```
=== Model Performance ===
Cross-validation AUC: 0.768 (77% accuracy in ranking)
Precision: 0.681 (68% of predictions are correct)  
Recall: 0.367 (captures 37% of actual participants)
F1-Score: 0.477

=== Conservative Threshold Selection ===
Selected threshold: 0.649 (85th percentile)
Total authors analyzed: 42,580
Authors predicted to participate: 1,669 (3.9%)
```

**Cross-Conference Summary:**
| Conference | Predictions | Matched Profiles | Match Rate | Avg H-Index | Avg Citations |
|------------|-------------|------------------|------------|-------------|---------------|
| AAAI 2026  | 1,669       | 1,528           | 91.6%      | 38.1        | 13,475        |
| NeurIPS 2025| 2,194      | 1,983           | 90.4%      | 41.1        | 19,505        |
| ICLR 2026  | 1,604       | 1,449           | 90.3%      | 43.3        | 22,927        |
| **TOTAL**  | **5,467**   | **4,960**       | **90.7%**  | **40.8**    | **18,636**    |

### � What Makes the Model Work

**1. 🎯 Temporal Feature Engineering**
```python
# Key features that drive predictions:
features = {
    'num_participations': 10.2,        # Strong predictor
    'years_since_last': 0.0,           # Recent activity critical  
    'participation_rate': 1.4,         # Papers per year
    'max_consecutive_years': 4.2,      # Consistency matters
    'exp_decay_sum': 2.8,             # Recency weighting
    'markov_prob': 0.65               # Momentum indicator
}
```

**2. 📈 Top Predictive Patterns**
- **💯 100% confidence predictions:** Authors with 15+ papers AND participation in last 2 years
- **🎯 High precision zone:** 85th percentile threshold captures highly active authors
- **📊 Conservative approach:** 3.9% participation rate aligns with historical conference acceptance rates

**3. 🏆 Predicted Author Characteristics**
```
✅ Average profile of predicted participants:
• Past participations: 9.8 papers
• Years since last: 0.0 (all recent participants)  
• Participation rate: 1.4 papers/year
• Recent participants (≤2 years): 100%
• Prolific authors (≥8 papers): 49.4%
```

### 🔍 Model Validation & Cross-Validation Strategy

**Why GroupKFold Cross-Validation:**
```python
# Prevents data leakage - same author never in train + test
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=author_names):
    # Train on authors A,B,C... test on authors X,Y,Z...
```

**Why 85th Percentile Threshold:**
- 🎯 **Historical accuracy:** ~4-6% of authors typically participate in subsequent conferences
- 📊 **Precision focus:** Better to predict 1,669 highly likely participants than 5,000 uncertain ones  
- 🔍 **Human review:** Conservative predictions are more actionable for downstream analysis

**Feature Importance Analysis:**
```
� Top features (from Gradient Boosting):
1. num_participations: Past activity is strongest signal
2. years_since_last: Recency matters more than career length
3. participation_rate: Consistent productivity indicator
4. max_consecutive_years: Sustained engagement patterns
5. exp_decay_sum: Recent activity weighted by recency
```

---

## 🛠️ Technical Implementation Details

### 🕷️ Web Scraping Strategy
```python
# Smart scraping: Fast requests + Selenium fallback
def extract_profile(driver, user_id):
    try:
        # Try fast requests + BeautifulSoup first
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
        else:
            raise Exception("Fallback to Selenium")
    except:
        # Use Selenium when requests blocked/fails
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
```
**Why this works:** Google Scholar can block automated requests. The hybrid approach keeps scraping fast (requests) but robust (Selenium fallback).

### 💾 Caching & Performance Optimization
```python
# Wikipedia lookup cache (joblib)
@lru_cache(maxsize=256)
def classify_researcher_from_summary(summary):
    return classifier(summary, researcher_labels)

# Fuzzy matching cache (persistent)
if cache_key in fuzzy_cache:
    return fuzzy_cache[cache_key]
fuzzy_cache[cache_key] = (match, score, institution)
joblib.dump(fuzzy_cache, FUZZY_CACHE_PATH)
```
**Performance impact:** 10x speedup on repeated runs. Wikipedia classification results cached permanently, fuzzy matches stored for profile linking.

### � Fuzzy Matching Algorithm
```python
# Name normalization + fuzzy matching
def clean_name(name):
    name = re.sub(r'[^\w\s]', ' ', name.lower())
    name = re.sub(r'\s+', ' ', name).strip()
    return name

# 80% similarity threshold for matching
match, score = process.extractOne(predicted_name, scholar_names, 
                                scorer=fuzz.token_sort_ratio)
if score >= 80:
    # Match found
```
**Why 80% threshold:** Balances precision vs recall. Too high (90%+) misses legitimate variations, too low (70%) includes false positives.

### � Feature Engineering Deep Dive
```python
def get_year_features(years, all_years):
    # Temporal recency with exponential decay
    exp_decay_sum = sum(np.exp(-0.5 * (max_year - y)) for y in years)
    
    # Markov transition probability  
    consecutive_pairs = sum(1 for i in range(1, len(years)) 
                          if years[i] == years[i-1] + 1)
    markov_prob = consecutive_pairs / (len(years) - 1)
    
    # Participation streaks
    streak = max_streak = 1
    for i in range(1, len(years)):
        if years[i] == years[i-1] + 1:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 1
```
**Why these features:** Academic careers have momentum. Recent activity (exp_decay), consistency (streaks), and transitions (markov_prob) are stronger predictors than simple counts.

### 🎯 Model Ensemble Design
```python
# Gradient boosting: handles feature interactions
gb = GradientBoostingClassifier(
    max_depth=3,           # Prevent overfitting  
    min_samples_split=20,  # Require sufficient samples
    learning_rate=0.05,    # Conservative learning
    n_estimators=100       # Balanced complexity
)

# Logistic regression: linear baseline
lr = LogisticRegression(
    C=0.1,                 # L2 regularization
    max_iter=5000,         # Ensure convergence
    penalty='l2'           # Ridge regularization
)

# Ensemble: combine strengths
ensemble = VotingClassifier([('gb', gb), ('lr', lr)], voting='soft')
calibrated = CalibratedClassifierCV(ensemble, method='isotonic')
```
**Why this ensemble:** GB captures complex patterns (author career trajectories), LR provides stable linear signal. Isotonic calibration ensures probabilities are meaningful for threshold selection.

---

## � Troubleshooting & FAQ

### 🔧 Common Issues & Solutions

**❌ "Profile not found" or 403 errors during scraping**
```bash
# Solution 1: Slower crawl rate
# Edit main.py: time.sleep(random.uniform(2.0, 5.0))

# Solution 2: Use different Chrome driver
# Install chromedriver: brew install chromedriver  

# Solution 3: Check Selenium setup
python -c "from selenium import webdriver; print('Selenium OK')"
```

**❌ Model training runtime warnings**
```bash
# Suppress numerical warnings (non-blocking)
python src/models/aaai_predict_authors.py --quiet-warnings

# Check for NaN/inf in features
python -c "import pandas as pd; df=pd.read_csv('data/raw/aaai25_papers_authors_split.csv'); print(df.describe())"
```

**❌ "FileNotFoundError: Could not find data at path"**  
```bash
# Check current directory
pwd
# Should be in Scholar_Scraper/ root

# Verify data files exist
ls data/raw/

# Use absolute paths if needed
python src/models/aaai_predict_authors.py --data-path /full/path/to/data.csv
```

**❌ Low fuzzy match rates (<80%)**
```bash
# Inspect fuzzy matching cache
python -c "import joblib; cache=joblib.load('cache/fuzzy_match_cache.joblib'); print(len(cache))"

# Lower threshold for more matches (trade precision for recall)
python src/conference_predictions_summary_2026.py --threshold 70

# Check name normalization
python -c "from src.conference_predictions_summary_2026 import clean_name; print(clean_name('John O\'Brien'))"
```

### 🔄 Reproducibility Checklist

**For reproducible experiments:**
```bash
# 1. Version control
git add data/processed/ 
git commit -m "Snapshot: scholar profiles v1.0"

# 2. Environment pinning  
pip freeze > requirements_exact.txt

# 3. Explicit paths
export AAAI_DATA_PATH=/absolute/path/to/data.csv
export AAAI_MODEL_PATH=/absolute/path/to/model.pkl
python src/models/aaai_predict_authors.py

# 4. Random seed (already set in code)
# random_state=42 in all sklearn models
```

### 🎯 Performance Optimization Tips

**For faster crawling:**
```python
# Parallel profile enrichment (already implemented)
BATCH_SIZE = 100  # Adjust based on memory
SAVE_INTERVAL = 100  # More frequent saves = less loss on crash

# Use SSD storage for caches
export CACHE_DIR=/path/to/fast/ssd/cache
```

**For faster predictions:**
```bash
# Use fewer cross-validation folds for quick experiments
# Edit models/*.py: gkf = GroupKFold(n_splits=3)  # Instead of 5

# Skip feature selection for speed
# Comment out: SelectKBest(f_classif, k=min(10, len(features)))
```

### � Data Quality Checks

**Validate your data before training:**
```python
import pandas as pd

# Check AAAI data quality
df = pd.read_csv('data/raw/aaai25_papers_authors_split.csv')
print(f"Records: {len(df)}")
print(f"Authors: {df['author'].nunique()}")  
print(f"Years: {sorted(df['year'].unique())}")
print(f"Missing authors: {df['author'].isna().sum()}")

# Check for reasonable participation rates
yearly_counts = df.groupby('year').size()
print(f"Papers per year: {yearly_counts}")
```

### 🌐 Institution Mapping Issues

**If institution matching is poor:**
```python
# Update domain_to_institution mapping in main.py
domain_to_institution.update({
    "yourcorp.com": "Your Corporation",  
    "youruniv.edu": "Your University"
})

# Check email extraction
df = pd.read_csv('data/processed/scholar_profiles.csv')
institutions = df['institution'].value_counts()
print(f"Top institutions: {institutions.head(10)}")
```

---

## 🧪 Testing & Development

### 🔬 Running Tests
```bash
# Run all tests
pytest -v

# Run specific test
pytest tests/test_path_resolution.py -v

# Test with coverage
pip install pytest-cov
pytest --cov=src tests/
```

### 🔧 Development Setup
```bash
# Development installation
git clone https://github.com/krishnanefx/Scholar_Scraper.git
cd Scholar_Scraper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black isort mypy

# Code formatting
black src/ tests/
isort src/ tests/

# Type checking  
mypy src/
```

### 🎯 Adding New Conferences

**To add a new conference (e.g., ICML):**
```python
# 1. Create scraper: src/scrapers/icml_scraper.py
class ICMLScraper:
    def scrape_papers(self, year):
        # Implement conference-specific scraping
        pass

# 2. Create prediction model: src/models/icml_predict_authors.py  
# Copy from aaai_predict_authors.py and adapt

# 3. Update summary script: src/conference_predictions_summary_2026.py
conferences = ['aaai', 'neurips', 'iclr', 'icml']  # Add icml

# 4. Add data files to data/raw/icml_*.csv
```

### 📊 Model Experimentation

**Try different models:**
```python
# In models/*.py, replace ensemble with:

# XGBoost (install: pip install xgboost)
from xgboost import XGBClassifier
model = XGBClassifier(random_state=42)

# LightGBM (install: pip install lightgbm)  
from lightgbm import LGBMClassifier
model = LGBMClassifier(random_state=42)

# Neural networks (install: pip install torch)
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
```

**Feature engineering experiments:**
```python
# Add new temporal features in get_year_features():
features.update({
    'participation_variance': np.var(year_gaps),
    'career_phase': (current_year - first_year) / 10,  # Early/mid/late career
    'recent_momentum': sum(1 for y in years if y >= current_year - 3),
    'coauthor_overlap': calculate_coauthor_network_strength(author, years)
})
```

---

## 📦 Requirements & Dependencies

### 🐍 Core Requirements
```
Python 3.8+ (tested on 3.9.6)
Chrome/Chromium browser (for Selenium fallback)
4GB+ RAM (for processing 42K+ profiles)
2GB+ disk space (for data + caches)
```

### 📚 Key Dependencies
```python
# Machine Learning
scikit-learn>=1.0.0     # Core ML pipeline
numpy>=1.20.0           # Numerical computing
pandas>=1.3.0           # Data manipulation

# Web Scraping  
requests>=2.25.0        # HTTP requests
beautifulsoup4>=4.9.0   # HTML parsing
selenium>=4.0.0         # Browser automation
fuzzywuzzy>=0.18.0      # String matching

# Data Processing
joblib>=1.1.0           # Model serialization + caching
aiohttp>=3.8.0          # Async HTTP for Wikipedia
mwparserfromhell>=0.6.0 # Wikipedia markup parsing

# Visualization
streamlit>=1.15.0       # Interactive dashboard
plotly>=5.0.0           # Interactive plots
matplotlib>=3.5.0       # Static plots

# NLP (optional but recommended)
transformers>=4.21.0    # Zero-shot classification
torch>=1.12.0           # Transformer models backend
```

### 🔧 Installation Notes
```bash
# macOS users might need:
brew install chromedriver

# Linux users might need:  
sudo apt-get install chromium-browser chromium-chromedriver

# Windows users: Download ChromeDriver manually
# https://chromedriver.chromium.org/
```

---

## 🤝 Contributing & Roadmap

### 🎯 Areas for Contribution

**🔬 Model Improvements:**
- Experiment with deep learning models (LSTM for temporal sequences)
- Add author network features (PageRank, centrality measures)
- Try multi-task learning (predict multiple conferences jointly)

**�️ Data Collection:**
- Add more conferences (ICML, ACL, EMNLP, CVPR, ICCV)
- Implement arXiv scraping for preprint analysis
- Add Semantic Scholar API integration

**📊 Analysis Features:**
- Conference recommendation system
- Author collaboration network analysis  
- Research trend prediction
- Institution ranking and analysis

**🎨 UI/UX:**
- Enhanced dashboard with interactive plots
- Real-time prediction updates
- Mobile-responsive interface
- API endpoint for external integration

### 🚀 Future Roadmap

**Short term (1-3 months):**
- [ ] Docker containerization for easy deployment
- [ ] CI/CD pipeline with GitHub Actions
- [ ] More comprehensive test suite
- [ ] Performance benchmarking suite

**Medium term (3-6 months):**
- [ ] Multi-year prediction capabilities
- [ ] Author influence scoring
- [ ] Conference acceptance rate prediction
- [ ] Research collaboration recommendations

**Long term (6+ months):**
- [ ] Real-time data streaming
- [ ] Integration with academic databases
- [ ] Conference organizer dashboard
- [ ] AI-powered research trend analysis

### 📄 License & Citation

**MIT License** - See LICENSE file for details

**If you use this work in research, please cite:**
```bibtex
@software{scholar_scraper_2025,
  title={Scholar Scraper: ML-Powered Academic Conference Participation Prediction},
  author={krishnanefx},
  year={2025},
  url={https://github.com/krishnanefx/Scholar_Scraper}
}
```

---

<div align="center">

### 🌟 Star this repo if you found it helpful! 🌟

[![GitHub stars](https://img.shields.io/github/stars/krishnanefx/Scholar_Scraper?style=social)](https://github.com/krishnanefx/Scholar_Scraper)
[![GitHub forks](https://img.shields.io/github/forks/krishnanefx/Scholar_Scraper?style=social)](https://github.com/krishnanefx/Scholar_Scraper)

**Made with ❤️ for the academic research community**

**👨‍💻 Author:** krishnan ([@krishnanefx](https://github.com/krishnanefx))

</div>
