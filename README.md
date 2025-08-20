# 🎓 Scholar Scraper

```
  ____       _           _             ____                                   
 / ___|  ___| |__   ___ | | __ _ _ __  / ___|  ___ _ __ __ _ _ __   ___ _ __     
 \___ \ / __| '_ \ / _ \| |/ _` | '__| \___ \ / __| '__/ _` | '_ \ / _ \ '__|    
  ___) | (__| | | | (_) | | (_| | |     ___) | (__| | | (_| | |_) |  __/ |       
 |____/ \___|_| |_|\___/|_|\__,_|_|    |____/ \___|_|  \__,_| .__/ \___|_|       
                                                            |_|                  
```

🚀 **Scholar Scraper** is a comprehensive Python-based tool designed to collect, process, and analyze academic profile data from major AI/ML conferences. The project includes advanced machine learning models for predicting conference participation, scholar profile matching, and interactive data exploration capabilities.

---

> 🎯 **Goal**: Predict next-year conference participation (computed dynamically as latest-year-in-data + 1) with conservative, high-precision predictions
> 
> 📊 **Scale**: 46K+ scholar profiles, ~5.5K predictions across AAAI/NeurIPS/ICLR, comprehensive academic insights

---

## ✨ Features

### 🔧 Core Capabilities
- **🕷️ Scholar Profile Scraping**: Automated collection of Google Scholar profiles with Wikipedia enrichment
- **📊 Conference Data Processing**: Aggregates participation data from AAAI, ICLR, and NeurIPS conferences
- **🤖 Machine Learning Predictions**: Advanced ensemble models predicting 2026 conference participation
- **🔍 Profile Matching**: Fuzzy string matching to link predicted participants with scholar profiles
- **📈 Interactive Dashboard**: Streamlit-based visualization and exploration platform

### 🧠 Advanced Analytics
- **🎯 Participation Prediction Models**: Gradient Boosting + Logistic Regression ensemble with 5-fold cross-validation
- **⚙️ Feature Engineering**: Temporal patterns, streak analysis, exponential decay scoring, Markov transitions
- **🎛️ Model Calibration**: Isotonic calibration with logic-based probability adjustments
- **👥 Comprehensive Profiling**: Institution mapping, research interest analysis, citation metrics

## 🗂️ Project Structure

### 📁 Core Files
- `📜 main.py`: Main script for scholar profile crawling and data processing
- `📊 dashboard.py`: Streamlit dashboard for data visualization and exploration
- `🎯 conference_predictions_summary_2026.py`: Comprehensive 2026 participant prediction with profile matching

### 🤖 Conference Prediction Models
- `AAAI scraper/predict_2026_authors.py`: AAAI 2026 participation prediction model
- `neurips_predict_2026_authors.py`: NeurIPS 2026 participation prediction model  
- `iclr_predict_2026_authors.py`: ICLR 2026 participation prediction model

### 💾 Data Files
- `👨‍🎓 scholar_profiles_progressssss.csv`: Complete scholar profile database (46K+ profiles)
- `📦 iclr_2020_2025_combined_data.parquet`: ICLR conference participation data (2020-2025)
- `📦 neurips_2020_2024_combined_data.parquet`: NeurIPS conference participation data (2020-2024)
- `📈 aaai_2026_predictions.csv`, `neurips_2026_predictions.csv`, `iclr_2026_predictions.csv`: 2026 predictions
- `📊 2026_Conference_Predictions_with_Scholar_Profiles.xlsx`: Comprehensive matched results

### ⚙️ Configuration & Cache
- `📋 queue.txt`: Profile crawling queue management
- `💾 wiki_lookup_cache.joblib`, `fuzzy_match_cache.json`: Performance optimization caches
- `📝 requirements.txt`: Python dependencies

## 🚀 Getting Started

### 📋 Prerequisites
- 🐍 Python 3.8+
- 🌐 Chrome/Chromium browser (for web scraping)
- 📦 Required Python packages (see `requirements.txt`)

### 💻 Installation
1. **📥 Clone the repository:**
   ```bash
   git clone https://github.com/krishnanefx/Scholar_Scraper.git
   cd Scholar_Scraper
   ```

2. **⚡ Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

> 💡 **Quick Start**: Follow these steps to get up and running in minutes!

#### 🕷️ Scholar Profile Scraping
```bash
🚀 python main.py
```
```
✨ What it does:
🔄 Crawls Google Scholar profiles using BFS traversal
📚 Enriches profiles with Wikipedia data
🔍 Performs fuzzy matching with conference participation data
```

#### 🤖 Conference Participation Prediction (CLI / env-friendly)

The model scripts now accept CLI flags and environment variables to make runs reproducible and workspace-independent. Each script will compute the prediction year as (max year in the input data) + 1 by default.

Examples:

```bash
# Use the bundled defaults (paths resolved relative to the script):
python src/models/aaai_predict_authors.py
python src/models/neurips_predict_authors.py
python src/models/iclr_predict_authors.py

# Or pass explicit paths (CLI)
python src/models/iclr_predict_authors.py \
   --data-path data/raw/iclr_2020_2025.parquet \
   --output-dir data/predictions \
   --model-path data/processed/iclr_participation_model.pkl

# Or set environment variables (ICLR example)
export ICLR_DATA_PATH=data/raw/iclr_2020_2025.parquet
export ICLR_OUTPUT_DIR=data/predictions
export ICLR_MODEL_PATH=data/processed/iclr_participation_model.pkl
python src/models/iclr_predict_authors.py
```

If training produces many numerical warnings from scikit-learn (divide-by-zero/overflow), they are non-blocking — use `--quiet-warnings` to suppress them during training runs.

## 🧪 Experiment runbook — how to run (copyable)

Quick commands to setup the environment and run model scripts reproducibly. Replace paths as needed.

1) Create & activate a virtualenv (macOS / zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Run model scripts (defaults resolve paths relative to script):

```bash
# Run ICLR prediction (auto computes prediction year = max_year + 1)
python src/models/iclr_predict_authors.py

# Run NeurIPS (example with explicit paths and quieter warnings)
python src/models/neurips_predict_authors.py \
   --data-path data/raw/neurips_2020_2024_combined_data.parquet \
   --output-dir data/predictions \
   --model-path data/processed/neurips_participation_model.pkl \
   --quiet-warnings

# Run AAAI
python src/models/aaai_predict_authors.py
```

3) Run using the venv's python explicitly (helpful in CI or scripts):

```bash
.venv/bin/python src/models/iclr_predict_authors.py --quiet-warnings
```

GPU / no-GPU guidance
- The prediction pipelines use scikit-learn (GradientBoostingClassifier + LogisticRegression) which run on CPU. No GPU required.
- If you plan to add GPU-accelerated training (e.g., LightGBM/XGBoost with GPU, or torch-based models), set CUDA and ensure GPU-enabled builds are installed. Example (bash):

```bash
export CUDA_VISIBLE_DEVICES=0
# then run your GPU-capable script (requires GPU-enabled packages)
python my_gpu_model.py
```

Reproducibility tips
- Pass explicit `--data-path` and `--model-path` when running experiments to keep outputs organized.
- Commit the repo and note the git SHA alongside any saved model files.


#### 📊 Comprehensive Analysis with Profile Matching
```bash
🎯 python conference_predictions_summary_2026.py
```
```
� Output:
📈 91.4% match rate with scholar profiles
� Comprehensive Excel report with full academic profiles
📋 Separate sheets for AAAI, NeurIPS, and ICLR predictions
```

#### 🎨 Interactive Dashboard
```bash
🌟 streamlit run dashboard.py
```
```
🔧 Features:
🔍 Explore scholar profiles and conference participation patterns
📊 Visualize prediction results and academic metrics
🎛️ Interactive filtering and analysis tools
```

## 🏆 Key Results (Summary)

The following consolidated results were generated from the historic conference datasets (AAAI, NeurIPS, ICLR). Predictions target the year after the last year available in each dataset (e.g., data to 2024 → predictions for 2025). The process uses conservative thresholding (85th percentile on training probabilities) to favor precision over recall.

| Conference | Predictions (selected) | Matched to Scholar Profiles | Match Rate | Avg H-Index | Avg Citations |
|------------|------------------------:|---------------------------:|----------:|------------:|--------------:|
| AAAI       | 1,669                  | 1,540                      | 92.3%     | 38.1        | 13,433        |
| NeurIPS    | 2,194                  | 1,999                      | 91.1%     | 41.0        | 19,312        |
| ICLR       | 1,625                  | 1,479                      | 91.0%     | 43.2        | 22,737        |
| TOTAL      | **5,488**              | **5,018**                  | **91.4%** | 40.8        | 18,494        |

### 🤖 Model Performance (detailed)
- Cross-validation AUC: ~0.76 - 0.85 across conferences (venue and dataset dependent)
- Cross-validation precision/recall/F1: conservative thresholding was chosen to keep precision high (~0.82-0.89) while maintaining moderate recall (~0.60-0.75) depending on conference and fold.
- Ensemble Method: VotingClassifier ensemble of Gradient Boosting (GBDT) and Logistic Regression, with isotonic calibration applied on the ensemble probabilities.
- Feature engineering highlights:
   - Lagged-year features: participation indicators per year (lagged to exclude target)
   - Streaks & gaps: consecutive-year streaks and gap statistics
   - Temporal recency: exponential decay sum for recent activity
   - Markov-like transitions: probability of participation given prior-year attendance
   - Rolling window trends: last-3-year rates and activity trend deltas

These features produced models that generalize well under GroupKFold cross-validation (grouped by author to avoid leakage).

### 🏅 Top Performing Institutions
```
🥇 Google LLC          🥈 Stanford University    🥉 Tsinghua University
🏆 MIT                 🏆 University of Toronto  🏆 Carnegie Mellon
```

## 🏗️ Technical Architecture

### 🤖 Machine Learning Pipeline
- **📊 Data Processing**: Temporal feature engineering with participation patterns
- **🎯 Model Architecture**: VotingClassifier ensemble (Gradient Boosting + Logistic Regression)
- **🔄 Cross-Validation**: 5-fold GroupKFold to prevent data leakage
- **📈 Calibration**: Isotonic calibration for probability estimation
- **🧠 Logic Integration**: Domain-specific probability adjustments

### 🔗 Data Integration
- **👨‍🎓 Scholar Profiles**: 46K+ profiles with Google Scholar and Wikipedia data
- **📚 Conference Data**: Historical participation from AAAI, ICLR, NeurIPS (2020-2025)
- **🔍 Fuzzy Matching**: 80% similarity threshold for name matching
- **✨ Profile Enrichment**: Institution mapping, research interest analysis, citation metrics

### ⚡ Performance Optimization
- **💾 Caching Systems**: Wikipedia lookup and fuzzy match caching
- **⚙️ Batch Processing**: Efficient handling of large-scale predictions
- **📋 Progress Tracking**: Queue management and incremental processing

## 📤 Output Formats

### 📊 Excel Reports
- **📈 2026_Conference_Predictions_with_Scholar_Profiles.xlsx**: Comprehensive matched results
- 📋 Individual sheets per conference with full academic profiles
- 📊 Summary statistics and unmatched entries for manual review

### 📄 CSV Files
- 📈 Individual prediction files for each conference
- 👨‍🎓 Scholar profile database with enriched metadata
- 💾 Cached matching results for performance optimization

## 📦 Requirements
- 🐍 Python 3.8+
- 🌐 Chrome/Chromium (for scraping components)
- Run `pip install -r requirements.txt` to install dependencies. The `requirements.txt` contains core ML/data libs and optional tooling for scraping and dashboarding.

Notable additions for developers:
- `pytest` — used for unit tests added under `tests/`
- `joblib` — for model serialization (used by scripts)

## 🤝 Contributing
🎉 Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

### 🔧 Development Setup
```bash
git clone https://github.com/krishnanefx/Scholar_Scraper.git
cd Scholar_Scraper
pip install -r requirements.txt
```

### 📋 TODO
- [ ] 🌐 Add more conferences (ICML, ACL, EMNLP)
- [ ] 🎨 Enhanced dashboard visualizations
- [ ] 📊 Real-time prediction updates
- [ ] 🔍 Advanced search and filtering

---

## 📄 License
📜 MIT License

## 👨‍💻 Author
**krishnan** (GitHub: [@krishnanefx](https://github.com/krishnanefx))

---

<div align="center">

### 🌟 Star this repo if you found it helpful! 🌟

[![GitHub stars](https://img.shields.io/github/stars/krishnanefx/Scholar_Scraper?style=social)](https://github.com/krishnanefx/Scholar_Scraper)
[![GitHub forks](https://img.shields.io/github/forks/krishnanefx/Scholar_Scraper?style=social)](https://github.com/krishnanefx/Scholar_Scraper)

**Made with ❤️ for the academic research community**

</div>
