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

# Create author features
author_years = df.groupby('author')['year'].apply(list).reset_index()
author_years_feats = author_years['year'].apply(lambda years: get_year_features(years, all_years))
author_feats_df = pd.DataFrame(list(author_years_feats))
author_full = pd.concat([author_years['author'], author_feats_df], axis=1)

print(f"Total authors in dataset: {len(author_full)}")

# CRITICAL FIX: Separate training and prediction datasets
# Training: Only authors with ≥2 participations (reliable patterns)
# Prediction: ALL authors (including newcomers)

# 1. Training data: Authors with sufficient history
reliable_authors = author_full[author_full['num_participations'] >= 2]['author'].tolist()
training_data = author_full[author_full['author'].isin(reliable_authors)].copy()

# Create target variable: did they participate in the most recent year?
training_data['appeared_target'] = training_data['author'].isin(
    df[df['year'] == max_year]['author']
).astype(int)

print(f"Training on {len(training_data)} authors with ≥2 participations")
print(f"Target distribution: {training_data['appeared_target'].value_counts().to_dict()}")

# 2. Cross-validation on training data
groups = training_data['author']
gkf = GroupKFold(n_splits=5)
results = []

print("\nStarting cross-validation...")
for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(training_data, training_data['appeared_target'], groups)):
    print(f"\nFold {fold_idx + 1}/5")
    
    X_train = training_data.iloc[train_idx][base_features].fillna(0)
    y_train = training_data.iloc[train_idx]['appeared_target']
    X_test = training_data.iloc[test_idx][base_features].fillna(0)
    y_test = training_data.iloc[test_idx]['appeared_target']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = clean_features(X_train_scaled)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = clean_features(X_test_scaled)
    
    # Feature selection
    selector = SelectKBest(f_classif, k=min(10, len(base_features)))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Conservative models
    gb = GradientBoostingClassifier(random_state=42, max_depth=3, min_samples_split=20, min_samples_leaf=10)
    lr = LogisticRegression(max_iter=5000, random_state=42, penalty='l2')
    
    # Grid search with conservative parameters
    gb_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.05]}
    lr_grid = {'C': [0.01, 0.1, 1]}
    
    gb_cv = GridSearchCV(gb, gb_grid, cv=3, n_jobs=-1, scoring='roc_auc')
    lr_cv = GridSearchCV(lr, lr_grid, cv=3, n_jobs=-1, scoring='roc_auc')
    
    gb_cv.fit(X_train_selected, y_train)
    lr_cv.fit(X_train_selected, y_train)
    
    # Ensemble
    ensemble = VotingClassifier(estimators=[
        ('gb', gb_cv.best_estimator_),
        ('lr', lr_cv.best_estimator_)
    ], voting='soft')
    
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
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"AUC: {auc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
        results.append({'auc': auc, 'precision': prec, 'recall': rec, 'f1': f1})
    else:
        print("Warning: Only one class in test set")
        results.append({'auc': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan})

# Overall CV results
valid_results = [r for r in results if not np.isnan(r['auc'])]
if valid_results:
    overall_auc = np.mean([r['auc'] for r in valid_results])
    overall_precision = np.mean([r['precision'] for r in valid_results]) 
    overall_recall = np.mean([r['recall'] for r in valid_results])
    overall_f1 = np.mean([r['f1'] for r in valid_results])
    
    print(f"\nOverall CV Results:")
    print(f"AUC: {overall_auc:.3f}")
    print(f"Precision: {overall_precision:.3f}")
    print(f"Recall: {overall_recall:.3f}")
    print(f"F1: {overall_f1:.3f}")
else:
    print("Warning: No valid CV results obtained")

print("\nTraining final model...")
# Train final model on all reliable authors
X_final = training_data[base_features].fillna(0)
y_final = training_data['appeared_target']

scaler_final = StandardScaler()
X_final_scaled = scaler_final.fit_transform(X_final)
X_final_scaled = clean_features(X_final_scaled)

selector_final = SelectKBest(f_classif, k=min(10, len(base_features)))
X_final_selected = selector_final.fit_transform(X_final_scaled, y_final)

# Final ensemble
gb_final = GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.05,
                                    max_depth=3, min_samples_split=20, min_samples_leaf=10)
lr_final = LogisticRegression(max_iter=5000, random_state=42, C=0.1, penalty='l2')

ensemble_final = VotingClassifier(estimators=[
    ('gb', gb_final),
    ('lr', lr_final)
], voting='soft')

ensemble_final.fit(X_final_selected, y_final)

# Apply calibration
calibrated_final = CalibratedClassifierCV(ensemble_final, method='isotonic', cv=3)
calibrated_final.fit(X_final_selected, y_final)

# 3. Generate predictions for ALL authors (including newcomers)
print(f"\nGenerating predictions for all {len(author_full)} authors...")

# Prepare features for prediction
X_all = author_full[base_features].fillna(0)
X_all_scaled = scaler_final.transform(X_all)
X_all_scaled = clean_features(X_all_scaled)
X_all_selected = selector_final.transform(X_all_scaled)

# Get raw model probabilities
raw_probabilities = calibrated_final.predict_proba(X_all_selected)[:, 1]

# CRITICAL: Apply logic-based probability adjustments
def adjust_probability_with_logic(row, raw_prob):
    """Apply domain knowledge to adjust probabilities"""
    num_participations = row['num_participations']
    years_since_last = row['years_since_last']
    
    # Base adjustments based on participation history
    if num_participations == 0:
        # Authors with no history: max 10% probability
        adjusted_prob = min(raw_prob, 0.10)
    elif num_participations == 1:
        # Authors with only 1 participation: max 30% probability  
        adjusted_prob = min(raw_prob, 0.30)
    else:
        # Authors with 2+ participations: use model probability
        adjusted_prob = raw_prob
    
    # Recency boost: +20% for authors active within last 2 years
    if years_since_last <= 2 and num_participations > 0:
        adjusted_prob = min(1.0, adjusted_prob * 1.2)
    
    return adjusted_prob

# Apply adjustments
author_full['raw_probability'] = raw_probabilities
author_full['adjusted_probability'] = author_full.apply(
    lambda row: adjust_probability_with_logic(row, row['raw_probability']), axis=1
)

# Use adjusted probabilities for final predictions
final_probabilities = author_full['adjusted_probability'].values

# Conservative threshold: Use 85th percentile of TRAINING authors only
training_probs = []
for idx, author in enumerate(author_full['author']):
    if author in reliable_authors:
        training_probs.append(final_probabilities[idx])

conservative_threshold = np.percentile(training_probs, 85)  # 85th percentile
final_predictions = (final_probabilities >= conservative_threshold).astype(int)

# Add results to dataframe
author_full['probability'] = final_probabilities
author_full['prediction'] = final_predictions
author_full['confidence'] = np.maximum(final_probabilities, 1 - final_probabilities)

print(f"\n=== AAAI {target_year} Predictions Complete ===")
print(f"Predictions saved to: aaai_2026_predictions.csv")
print(f"Total authors analyzed: {len(author_full)}")
print(f"Authors predicted to participate: {final_predictions.sum()}")
print(f"Conservative threshold: {conservative_threshold:.3f}")

# Get top predictions
top_predictions = author_full.nlargest(10, 'probability')
print(f"\nTop 10 most likely to participate:")
for _, row in top_predictions.iterrows():
    print(f"{row['author']}: {row['probability']:.3f} ({row['probability']*100:.1f}% confidence, {int(row['num_participations'])} past participations)")

# Match with scholar profiles if available
if scholar_profiles is not None:
    print(f"\nMatching with scholar profiles...")
    
    # Normalize names for matching
    def normalize_for_matching(name):
        if pd.isnull(name):
            return ''
        return re.sub(r'[^a-z\s]', '', name.lower().strip())
    
    author_full['normalized_name'] = author_full['author'].apply(normalize_for_matching)
    scholar_profiles['normalized_name'] = scholar_profiles['name'].apply(normalize_for_matching)
    
    # Merge with scholar profiles
    available_columns = ['normalized_name', 'name']
    
    # Check which optional columns exist in scholar profiles
    optional_columns = ['institution', 'email', 'research_interests']
    for col in optional_columns:
        if col in scholar_profiles.columns:
            available_columns.append(col)
    
    matched_df = author_full.merge(
        scholar_profiles[available_columns],
        on='normalized_name',
        how='left'
    )
    
    # Count matches
    matched_count = matched_df['name'].notna().sum()
    print(f"Successfully matched {matched_count}/{len(author_full)} authors with scholar profiles")
    
    # Use scholar profile data for final output
    base_columns = ['author', 'prediction', 'probability', 'num_participations', 'years_since_last', 'name']
    optional_columns = []
    
    if 'institution' in matched_df.columns:
        optional_columns.append('institution')
    if 'email' in matched_df.columns:
        optional_columns.append('email')
    if 'research_interests' in matched_df.columns:
        optional_columns.append('research_interests')
    
    final_output = matched_df[base_columns + optional_columns].copy()
    
    # Clean up column names
    rename_dict = {
        'author': 'AAAI_Author_Name',
        'name': 'Scholar_Profile_Name'
    }
    
    if 'institution' in final_output.columns:
        rename_dict['institution'] = 'Institution'
    if 'email' in final_output.columns:
        rename_dict['email'] = 'Email'
    if 'research_interests' in final_output.columns:
        rename_dict['research_interests'] = 'Research_Interests'
    
    final_output = final_output.rename(columns=rename_dict)
else:
    # No scholar profiles available
    final_output = author_full[[
        'author', 'prediction', 'probability', 'num_participations', 'years_since_last'
    ]].copy()
    final_output = final_output.rename(columns={'author': 'AAAI_Author_Name'})

# Sort by probability descending
final_output = final_output.sort_values('probability', ascending=False)
final_output['rank'] = range(1, len(final_output) + 1)

# Save results
final_output.to_csv('aaai_2026_predictions.csv', index=False)

# Save model
joblib.dump(calibrated_final, 'aaai_participation_model.pkl')
print(f"\nModel saved as: aaai_participation_model.pkl")

print("Done!")

# Clean up the rest of the file by removing unused code

