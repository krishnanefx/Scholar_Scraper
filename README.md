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

> 🎯 **Goal**: Predict 2026 conference participation with 91.4% accuracy across AAAI, NeurIPS, and ICLR
> 
> 📊 **Scale**: 46K+ scholar profiles, 5K+ predictions, comprehensive academic insights

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

#### 🤖 Conference Participation Prediction
```bash
# 🎯 Generate 2026 predictions for all conferences
🟡 python AAAI\ scraper/predict_2026_authors.py
🔵 python neurips_predict_2026_authors.py  
🟢 python iclr_predict_2026_authors.py
```

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

## 🏆 Key Results

| �️ Conference | 🎯 Predictions | ✅ Matched | 📊 Match Rate | 📈 Avg H-Index | 📚 Avg Citations |
|---------------|---------------|-----------|-------------|---------------|-----------------|
| 🟡 **AAAI 2026** | 1,669 | 1,540 | **92.3%** | 38.1 | 13,433 |
| 🔵 **NeurIPS 2026** | 2,194 | 1,999 | **91.1%** | 41.0 | 19,312 |
| 🟢 **ICLR 2026** | 1,625 | 1,479 | **91.0%** | 43.2 | 22,737 |
| 🎉 **TOTAL** | **5,488** | **5,018** | **91.4%** | 40.8 | 18,494 |

### 🤖 Model Performance
- **📊 Cross-validation AUC**: 0.76-0.80 across conferences
- **⚙️ Feature Engineering**: Temporal patterns, participation streaks, Markov transitions
- **🎯 Ensemble Method**: Gradient Boosting + Logistic Regression with isotonic calibration

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
- 🌐 Chrome/Chromium browser
- 📝 See `requirements.txt` for complete dependency list

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
