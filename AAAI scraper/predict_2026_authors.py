#!/usr/bin/env python3
"""
AAAI 2026 Author Participation Prediction Model
Consistent with NeurIPS and ICLR prediction models
"""

from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import calibration_curve
from collections import defaultdict, Counter
import re
import matplotlib.pyplot as plt
import joblib
import os

def clean_features(X):
    """Replace NaN, inf, -inf with 0 and clip extreme values."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    return X

def normalize_author(name):
    """Clean and normalize author names."""
    if pd.isnull(name):
        return ''
    name = name.strip()
    name = re.sub(r'\s+', ' ', name)
    name = name.lower()
    name = name.replace('.', '')
    name = re.sub(r'[^a-z\s]', '', name)
    return name.title()

def get_year_features(years, all_years):
    """Extract temporal features from author's participation history."""
    years_set = set(years)
    # Participation in each year (lagged: only up to t-1)
    year_feats = {f'appeared_{y}': int(y in years_set) for y in all_years}
    # Only use years < max(all_years) for lagged features
    past_years_attended = sorted([y for y in years if y in all_years])
    
    # Streak features (lagged)
    streak = 1
    max_streak = 1
    for i in range(1, len(past_years_attended)):
        if past_years_attended[i] == past_years_attended[i-1] + 1:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 1
    
    # Years since first/last participation (lagged)
    if past_years_attended:
        last_year = past_years_attended[-1]
        first_year = past_years_attended[0]
        years_since_last = max(all_years) - last_year
        years_since_first = max(all_years) - first_year
        
        # Participation rate
        career_span = last_year - first_year + 1 if last_year > first_year else 1
        participation_rate = len(past_years_attended) / career_span if career_span > 0 else 0
    else:
        years_since_last = len(all_years)
        years_since_first = len(all_years)
        participation_rate = 0
    
    # Gap between participations
    gaps = []
    for i in range(1, len(past_years_attended)):
        gaps.append(past_years_attended[i] - past_years_attended[i-1] - 1)
    
    # Exponential decay sum (recency weighting)
    exp_decay_sum = sum(np.exp(-0.5 * (max(all_years) - y)) for y in past_years_attended)
    
    # Markov-like transition probabilities
    consecutive_pairs = 0
    total_pairs = len(past_years_attended) - 1
    for i in range(1, len(past_years_attended)):
        if past_years_attended[i] == past_years_attended[i-1] + 1:
            consecutive_pairs += 1
    markov_prob = consecutive_pairs / total_pairs if total_pairs > 0 else 0
    
    features = {
        'num_participations': len(past_years_attended),
        'max_consecutive_years': max_streak,
        'years_since_last': years_since_last,
        'years_since_first': years_since_first,
        'participation_rate': participation_rate,
        'max_gap': max(gaps) if gaps else 0,
        'mean_gap': np.mean(gaps) if gaps else 0,
        'std_gap': np.std(gaps) if gaps else 0,
        'exp_decay_sum': exp_decay_sum,
        'markov_prob': markov_prob,
    }
    
    features.update(year_feats)
    return features

def adjust_probability_with_logic(row, raw_prob):
    """Apply domain knowledge to adjust probabilities logically."""
    num_participations = row['num_participations']
    years_since_last = row['years_since_last']
    
    # Conservative adjustments based on participation history
    if num_participations == 0:
        # Authors with no history: very low probability
        adjusted_prob = min(raw_prob, 0.10)
    elif num_participations == 1:
        # Authors with single participation: moderate cap
        adjusted_prob = min(raw_prob, 0.30)
    else:
        # Authors with 2+ participations: use model probability
        adjusted_prob = raw_prob
    
    # Recency boost
    if years_since_last <= 2 and num_participations > 0:
        adjusted_prob = min(1.0, adjusted_prob * 1.2)
    
    return adjusted_prob

# Load the data
print("Loading AAAI data...")
print(f"Current working directory: {os.getcwd()}")

if os.path.exists('aaai25_papers_authors_split.csv'):
    df = pd.read_csv('aaai25_papers_authors_split.csv')
elif os.path.exists('AAAI scraper/aaai25_papers_authors_split.csv'):
    df = pd.read_csv('AAAI scraper/aaai25_papers_authors_split.csv')
else:
    print("Error: Could not find aaai25_papers_authors_split.csv")
    exit(1)

# Data preprocessing
df['author'] = df['author'].fillna('').apply(normalize_author)
df = df[df['author'] != '']
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

print(f"Data loaded: {len(df)} paper-author pairs")
print(f"Unique authors: {df['author'].nunique()}")
print(f"Years covered: {df['year'].min()} to {df['year'].max()}")

# Years setup
all_years_full = sorted(df['year'].unique())
min_year, max_year = min(all_years_full), max(all_years_full)
target_year = max_year + 1  # Predicting 2026

# Use all years except target for features
all_years = all_years_full[:-1] if len(all_years_full) > 1 else all_years_full
print(f"Feature engineering using years: {min(all_years)} to {max(all_years)}")
print(f"Predicting participation for: {target_year}")

# Feature engineering
print("Extracting features...")
author_years = df.groupby('author')['year'].apply(list).reset_index()
author_features = []

for _, row in author_years.iterrows():
    features = get_year_features(row['year'], all_years)
    features['author'] = row['author']
    author_features.append(features)

# Create feature matrix
author_df = pd.DataFrame(author_features)
print(f"Features extracted for {len(author_df)} authors")

# CRITICAL: Separate training and prediction datasets
# Training: Only authors with ≥2 participations for reliable patterns
# Prediction: ALL authors (including newcomers)

training_authors = author_df[author_df['num_participations'] >= 2]['author'].tolist()
training_data = author_df[author_df['author'].isin(training_authors)].copy()

# Create target: did they participate in the most recent year?
training_data['target'] = training_data['author'].isin(
    df[df['year'] == max_year]['author']
).astype(int)

print(f"Training dataset: {len(training_data)} authors with ≥2 participations")
print(f"Target distribution: {training_data['target'].value_counts().to_dict()}")

# Features for modeling
feature_cols = [col for col in training_data.columns if col not in ['author', 'target']]
print(f"Using {len(feature_cols)} features")

# Cross-validation
groups = training_data['author']
gkf = GroupKFold(n_splits=5)
cv_results = []

print("\nStarting cross-validation...")
for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(training_data, training_data['target'], groups)):
    print(f"\nFold {fold_idx + 1}/5")
    
    X_train = training_data.iloc[train_idx][feature_cols].fillna(0)
    y_train = training_data.iloc[train_idx]['target']
    X_test = training_data.iloc[test_idx][feature_cols].fillna(0)
    y_test = training_data.iloc[test_idx]['target']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = clean_features(X_train_scaled)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = clean_features(X_test_scaled)
    
    # Feature selection
    selector = SelectKBest(f_classif, k=min(15, len(feature_cols)))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Models with conservative parameters
    gb = GradientBoostingClassifier(
        random_state=42, 
        max_depth=3, 
        min_samples_split=20, 
        min_samples_leaf=10,
        n_estimators=100,
        learning_rate=0.05
    )
    
    lr = LogisticRegression(
        max_iter=5000, 
        random_state=42, 
        penalty='l2',
        C=0.1
    )
    
    # Ensemble
    ensemble = VotingClassifier(
        estimators=[('gb', gb), ('lr', lr)], 
        voting='soft'
    )
    
    ensemble.fit(X_train_selected, y_train)
    
    # Calibration
    calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
    calibrated.fit(X_train_selected, y_train)
    
    # Predictions
    y_pred = calibrated.predict(X_test_selected)
    y_proba = calibrated.predict_proba(X_test_selected)[:, 1]
    
    # Metrics
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"AUC: {auc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        cv_results.append({'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1})
    else:
        print("Warning: Only one class in test set")
        cv_results.append({'auc': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan})

# Overall CV results
valid_results = [r for r in cv_results if not np.isnan(r['auc'])]
if valid_results:
    print(f"\nOverall CV Results:")
    print(f"AUC: {np.mean([r['auc'] for r in valid_results]):.3f}")
    print(f"Precision: {np.mean([r['precision'] for r in valid_results]):.3f}")
    print(f"Recall: {np.mean([r['recall'] for r in valid_results]):.3f}")
    print(f"F1: {np.mean([r['f1'] for r in valid_results]):.3f}")

# Train final model
print("\nTraining final model...")
X_final = training_data[feature_cols].fillna(0)
y_final = training_data['target']

scaler_final = StandardScaler()
X_final_scaled = scaler_final.fit_transform(X_final)
X_final_scaled = clean_features(X_final_scaled)

selector_final = SelectKBest(f_classif, k=min(15, len(feature_cols)))
X_final_selected = selector_final.fit_transform(X_final_scaled, y_final)

# Final ensemble
gb_final = GradientBoostingClassifier(
    random_state=42, 
    n_estimators=100, 
    learning_rate=0.05,
    max_depth=3, 
    min_samples_split=20, 
    min_samples_leaf=10
)

lr_final = LogisticRegression(
    max_iter=5000, 
    random_state=42, 
    C=0.1, 
    penalty='l2'
)

ensemble_final = VotingClassifier(
    estimators=[('gb', gb_final), ('lr', lr_final)], 
    voting='soft'
)

ensemble_final.fit(X_final_selected, y_final)

# Apply calibration
calibrated_final = CalibratedClassifierCV(ensemble_final, method='isotonic', cv=3)
calibrated_final.fit(X_final_selected, y_final)

# Generate predictions for ALL authors
print(f"\nGenerating predictions for all {len(author_df)} authors...")

X_all = author_df[feature_cols].fillna(0)
X_all_scaled = scaler_final.transform(X_all)
X_all_scaled = clean_features(X_all_scaled)
X_all_selected = selector_final.transform(X_all_scaled)

# Get raw probabilities
raw_probabilities = calibrated_final.predict_proba(X_all_selected)[:, 1]

# Apply logic-based adjustments
author_df['raw_probability'] = raw_probabilities
author_df['adjusted_probability'] = author_df.apply(
    lambda row: adjust_probability_with_logic(row, row['raw_probability']), axis=1
)

final_probabilities = author_df['adjusted_probability'].values

# Conservative threshold: Use 85th percentile of TRAINING authors only
training_probs = []
for idx, author in enumerate(author_df['author']):
    if author in training_authors:
        training_probs.append(final_probabilities[idx])

conservative_threshold = np.percentile(training_probs, 85) if training_probs else 0.5
final_predictions = (final_probabilities >= conservative_threshold).astype(int)

# Prepare final output in consistent format
results = []
for idx, row in author_df.iterrows():
    results.append({
        'predicted_author': row['author'],
        'will_participate_2026': final_predictions[idx],
        'participation_probability': final_probabilities[idx],
        'confidence_percent': final_probabilities[idx] * 100,
        'num_participations': int(row['num_participations']),
        'years_since_last': int(row['years_since_last']),
        'participation_rate': row['participation_rate'],
        'rank': 0  # Will be set after sorting
    })

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('participation_probability', ascending=False)
results_df['rank'] = range(1, len(results_df) + 1)

# Save results to correct location
output_path = 'aaai_2026_predictions.csv'
if not os.path.exists('aaai25_papers_authors_split.csv'):
    # We're running from parent directory
    output_path = 'AAAI scraper/aaai_2026_predictions.csv'

results_df.to_csv(output_path, index=False)

# Save model
model_path = 'aaai_participation_model.pkl'
if not os.path.exists('aaai25_papers_authors_split.csv'):
    model_path = 'AAAI scraper/aaai_participation_model.pkl'

joblib.dump(calibrated_final, model_path)

# Print summary
print(f"\n=== AAAI {target_year} Predictions Complete ===")
print(f"Total authors analyzed: {len(results_df)}")
print(f"Authors predicted to participate: {final_predictions.sum()}")
print(f"Conservative threshold: {conservative_threshold:.3f}")

# Top predictions
predicted_participants = results_df[results_df['will_participate_2026'] == 1]
print(f"\nTop 10 most likely to participate:")
for _, row in predicted_participants.head(10).iterrows():
    print(f"{row['predicted_author']}: {row['participation_probability']:.3f} "
          f"({row['confidence_percent']:.1f}% confidence, {row['num_participations']} past participations)")

print(f"\nPredictions saved to: {output_path}")
print(f"Model saved as: {model_path}")
print("Done!")
