# 🎓 Scholar Scraper: ML-Powered Academic Conference Participation Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**A production-ready system for predicting future conference participation using ensemble machine learning and temporal feature engineering.**

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Scholar Scraper is a comprehensive academic analytics platform that predicts future conference participation across major AI/ML venues (AAAI, NeurIPS, ICLR) using advanced machine learning techniques. The system achieves state-of-the-art performance through temporal feature engineering and ensemble methods.

### Key Capabilities
- **Predictive Modeling**: Forecast conference participation with high accuracy
- **Profile Enrichment**: Match predictions to comprehensive scholar profiles
- **Interactive Dashboard**: Explore 60,000 researcher profiles with advanced filtering
- **Network Analysis**: Visualize collaboration networks with Gephi export
- **Cross-Conference Analysis**: Unified framework for comparative analysis

## ✨ Features

### 🔮 Prediction Engine
- **Temporal Features**: Exponential decay weighting, Markov transition probabilities
- **Ensemble Methods**: Gradient Boosting + Logistic Regression with soft voting
- **Cross-Validation**: Rigorous GroupKFold validation preventing data leakage
- **Conservative Thresholding**: 85th percentile probability adjustment

### 📊 Interactive Dashboard
- **Profile Browser**: Search and filter 60,000 scholar profiles
- **Analytics Dashboard**: H-index distributions, geographic analysis, awards tracking
- **Network Visualization**: Author collaboration networks with comprehensive labeling
- **Singaporean Co-authors Analysis**: Specialized analysis for Singapore research ecosystem

### 🔗 Data Integration
- **Profile Matching**: 91% success rate linking predictions to academic profiles
- **Export Options**: CSV, JSON, Excel formats with fuzzy matching
- **Gephi Integration**: Network analysis with .gexf export for advanced visualization

## �️ Installation

```bash
# Clone the repository
git clone https://github.com/krishnanefx/Scholar_Scraper.git
cd Scholar_Scraper

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Generate Predictions (5 Minutes)
```bash
# Run prediction pipeline for all conferences
python src/models/aaai_predict_authors.py --quiet-warnings
python src/models/neurips_predict_authors.py --quiet-warnings
python src/models/iclr_predict_authors.py --quiet-warnings

# Create comprehensive Excel report
python src/conference_predictions_summary_2026.py --year 2026

# Launch interactive dashboard
streamlit run src/dashboard.py
```

### What You'll Get
- **5,875 predictions** across AAAI 2026, NeurIPS 2025, ICLR 2026
- **91% profile matching** success rate
- **Interactive dashboard** for data exploration
- **Excel reports** with institutional affiliations and metrics

## 📖 Usage

### Prediction Scripts
```bash
# Generate predictions for specific conference
python src/models/aaai_predict_authors.py --quiet-warnings
python src/models/neurips_predict_authors.py --quiet-warnings
python src/models/iclr_predict_authors.py --quiet-warnings
```

### Dashboard
```bash
# Launch interactive dashboard
streamlit run src/dashboard.py
```

### Advanced Options
```bash
# Custom prediction year and threshold
python src/conference_predictions_summary_2026.py --year 2027 --threshold 80

# Feature selection for AAAI
python src/models/aaai_predict_authors.py --feature-selection tree
```

## 📁 Project Structure

```
Scholar_Scraper/
├── src/
│   ├── models/           # Prediction scripts
│   │   ├── aaai_predict_authors.py
│   │   ├── neurips_predict_authors.py
│   │   └── iclr_predict_authors.py
│   ├── scrapers/         # Data collection tools
│   ├── dashboard.py      # Streamlit application
│   └── conference_predictions_summary_2026.py
├── data/
│   ├── raw/             # Input datasets
│   ├── processed/       # Processed data and models
│   └── predictions/     # Prediction outputs
├── outputs/             # Excel reports and visualizations
├── requirements.txt     # Python dependencies
└── README.md
```

## 📊 Performance

### Model Performance Summary

| Conference | Authors | Predictions | AUC | Precision | Recall | F1-Score | Match Rate |
|------------|---------|-------------|-----|-----------|--------|----------|------------|
| **AAAI 2026** | 42,580 | 2,108 (5.0%) | **0.861** | **0.894** | 0.629 | 0.739 | **91.6%** |
| **ICLR 2026** | 27,836 | 1,604 (5.8%) | **0.863** | 0.873 | **0.778** | **0.822** | **90.3%** |
| **NeurIPS 2025** | 35,705 | 2,194 (6.1%) | 0.840 | 0.875 | 0.693 | 0.773 | **90.4%** |

### Key Metrics Explained
- **AUC**: Model's ability to distinguish participants from non-participants
- **Precision**: Accuracy of positive predictions (fewer false positives)
- **Recall**: Coverage of actual participants (fewer missed predictions)
- **F1-Score**: Balanced measure of precision and recall

## 📄 License

This project is licensed under the MIT License 
## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the interactive dashboard
- Uses [scikit-learn](https://scikit-learn.org/) for machine learning
- Network visualization powered by [NetworkX](https://networkx.org/) and [Matplotlib](https://matplotlib.org/)

## 📞 Contact

**Author**: Krishnan   
**Email**: krish.adaik@gmail.com  
**GitHub**: [@krishnanefx](https://github.com/krishnanefx)

---

<div align="center">

### 🌟 Star this repository if it helps your research! 🌟

[![GitHub stars](https://img.shields.io/github/stars/krishnanefx/Scholar_Scraper?style=social)](https://github.com/krishnanefx/Scholar_Scraper)
[![GitHub forks](https://img.shields.io/github/forks/krishnanefx/Scholar_Scraper?style=social)](https://github.com/krishnanefx/Scholar_Scraper)

**🎓 Advancing Academic Intelligence Through Machine Learning**

</div>
