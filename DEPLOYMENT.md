# Streamlit Cloud Deployment Guide

## 🚀 Deploying Your Scholar Scraper Dashboard

### Prerequisites
1. Your repository is on GitHub
2. You have a Streamlit Cloud account (https://streamlit.io/)
3. The `data/processed/scholar_profiles.csv` file exists in your repository

### Step-by-Step Deployment

#### 1. Prepare Your Repository
Make sure these files are in your repository:
```
Scholar_Scraper/
├── src/
│   └── dashboard.py
├── data/
│   └── processed/
│       └── scholar_profiles.csv  # THIS IS CRITICAL!
├── requirements.txt
└── README.md
```

#### 2. Check File Sizes
- Streamlit Cloud has file size limits
- If your `scholar_profiles.csv` is too large (>100MB), consider:
  - Compressing it to `.csv.gz`
  - Splitting it into smaller files
  - Using a sample dataset for demo

#### 3. Deploy on Streamlit Cloud

1. **Go to:** https://share.streamlit.io/
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Fill in the details:**
   - Repository: `krishnanefx/Scholar_Scraper`
   - Branch: `main` 
   - Main file path: `src/dashboard.py`
5. **Click "Deploy!"**

#### 4. Troubleshooting Common Issues

**❌ "No profiles file found"**
- Solution: Make sure `data/processed/scholar_profiles.csv` exists in your GitHub repo
- Check that it's not in `.gitignore`
- Verify the file was committed and pushed

**❌ "File too large"**
- Solution: Use Git LFS for large files:
  ```bash
  git lfs track "*.csv"
  git add .gitattributes
  git add data/processed/scholar_profiles.csv
  git commit -m "Add profiles data with LFS"
  ```

**❌ "Module not found"**
- Solution: Update `requirements.txt` with all dependencies
- Make sure all imports in `dashboard.py` are included

#### 5. App Configuration (Optional)

Create `.streamlit/config.toml` in your repo:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
maxUploadSize = 200
maxMessageSize = 200
```

### 🎯 Quick Test

Before deploying, test locally:
```bash
cd Scholar_Scraper/src
streamlit run dashboard.py
```

### 📝 Notes for Your Specific Setup

1. **Your dashboard path:** `src/dashboard.py`
2. **Your data file:** `data/processed/scholar_profiles.csv` (32MB)
3. **Current file structure:** ✅ Correct
4. **Requirements:** ✅ Updated

### 🔧 If You Need a Demo Version

If your data file is too large or private, you can create a demo version:

1. **Create a smaller sample:**
   ```python
   import pandas as pd
   df = pd.read_csv('data/processed/scholar_profiles.csv')
   sample_df = df.sample(n=1000)  # Take 1000 random profiles
   sample_df.to_csv('data/processed/demo_profiles.csv', index=False)
   ```

2. **Update the dashboard** to use demo data for deployment:
   ```python
   # In dashboard.py, modify get_profiles_file_path()
   # Add demo file as fallback option
   ```

### 🚀 Your App URL
Once deployed, your app will be available at:
`https://krishnanefx-scholar-scraper-src-dashboard-xxxxx.streamlit.app/`
