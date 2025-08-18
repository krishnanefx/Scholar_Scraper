#!/usr/bin/env python3
"""
NeurIPS 2026 Author Participation Prediction Model
Adapted from AAAI prediction model to work with NeurIPS dataset
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
    
    # Lagged years since first/last participation (exclude target year)
    if past_years_attended:
        last_year = past_years_attended[-1]
        first_year = past_years_attended[0]
        years_since_last = max(all_years) - last_year
        years_since_first = max(all_years) - first_year
    else:
        last_year = first_year = years_since_last = years_since_first = 0
    
    # Participation rate (lagged)
    active_years = last_year - first_year + 1 if last_year > first_year else 1
    participation_rate = len(past_years_attended) / active_years if active_years > 0 else 0
    
    # Gap features (lagged)
    gaps = [past_years_attended[i] - past_years_attended[i-1] - 1 for i in range(1, len(past_years_attended))]
    max_gap = max(gaps) if gaps else 0
    mean_gap = np.mean(gaps) if gaps else 0
    
    # Temporal recency: exponential decay sum (lagged)
    decay = 0.5
    exp_decay_sum = sum(np.exp(-decay * (max(all_years) - y)) for y in past_years_attended)
    
    # Markov: probability of participating given previous year (lagged)
    markov_prob = 0.0
    if len(past_years_attended) > 1:
        transitions_count = 0
        for i in range(1, len(past_years_attended)):
            if past_years_attended[i] == past_years_attended[i-1] + 1:
                transitions_count += 1
        markov_prob = transitions_count / (len(past_years_attended) - 1)
    
    # Rolling window features for trend detection (last 3 years)
    recent_years = [y for y in past_years_attended if y >= max(all_years) - 2] if max(all_years) >= 3 else past_years_attended
    recent_participation_rate = len(recent_years) / 3 if len(recent_years) <= 3 else len(recent_years) / len(recent_years)
    
    # Activity trend (increasing/decreasing participation)
    activity_trend = 0
    if len(past_years_attended) >= 3:
        recent_3 = len([y for y in past_years_attended if y >= max(all_years) - 2])
        older_3 = len([y for y in past_years_attended if max(all_years) - 5 <= y < max(all_years) - 2])
        if older_3 > 0:
            activity_trend = (recent_3 - older_3) / older_3
    
    career_length = last_year - first_year + 1 if last_year >= first_year else 1
    normalized_participation_rate = len(past_years_attended) / career_length if career_length > 0 else 0
    
    return {
        'num_participations': len(past_years_attended),
        'last_year': last_year,
        'first_year': first_year,
        'career_length': career_length,
        'normalized_participation_rate': normalized_participation_rate,
        'recent_participation_rate': recent_participation_rate,
        'activity_trend': activity_trend,
        'max_consecutive_years': max_streak,
        'years_since_last': years_since_last,
        'years_since_first': years_since_first,
        'exp_decay_sum': exp_decay_sum,
        'markov_prob': markov_prob,
        'participation_rate': participation_rate,
        'max_gap': max_gap,
        'mean_gap': mean_gap,
        'streak_length': max_streak,
        **year_feats
    }

def main():
    print("=== NeurIPS 2026 Author Participation Prediction ===")
    
        # Load NeurIPS data
    print("Loading NeurIPS data...")
    df = pd.read_parquet('../../data/raw/neurips_2020_2024_combined_data.parquet')
    print(f"Loaded {len(df)} records from NeurIPS parquet dataset")
    
    # Data Quality: Clean and normalize author names
    df['Author'] = df['Author'].fillna('').apply(normalize_author)
    df = df[df['Author'] != '']
    
    # Handle missing/inconsistent years
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    
    print(f"After cleaning: {len(df)} records")
    print(f"Years available: {sorted(df['Year'].unique())}")
    print(f"Unique authors: {df['Author'].nunique()}")
    
    # Feature engineering
    all_years_full = sorted(df['Year'].unique())
    min_year, max_year = min(all_years_full), max(all_years_full)
    target_year = max_year
    all_years = [y for y in all_years_full if y < target_year]
    
    base_features = [
        'num_participations', 'max_consecutive_years', 'years_since_last',
        'years_since_first', 'exp_decay_sum', 'markov_prob', 'participation_rate',
        'max_gap', 'mean_gap', 'streak_length', 'career_length',
        'normalized_participation_rate', 'recent_participation_rate', 'activity_trend'
    ]
    
    # Create features for ALL authors first
    author_years = df.groupby('Author')['Year'].apply(list).reset_index()
    author_years_feats = author_years['Year'].apply(lambda years: get_year_features(years, all_years))
    author_feats_df = pd.DataFrame(list(author_years_feats))
    author_full = pd.concat([author_years['Author'], author_feats_df], axis=1)
    
    # Filter out authors with insufficient history for TRAINING only (min 2 participations)
    author_participation_counts = df.groupby('Author').size()
    reliable_authors = author_participation_counts[author_participation_counts >= 2].index
    training_mask = author_full['Author'].isin(reliable_authors)
    author_training = author_full[training_mask].copy()
    print(f"Training on {len(author_training)} authors (≥2 participations)")
    print(f"Total authors for prediction: {len(author_full)}")
    
    # Create target variable (only for training authors)
    target_year_authors = set(df[df['Year'] == target_year]['Author'])
    author_training['appeared_target'] = author_training['Author'].isin(target_year_authors).astype(int)
    
    print(f"Target distribution in training set: {np.bincount(author_training['appeared_target'])}")
    
    # Cross-validation with GroupKFold (using training set only)
    groups = author_training['Author']
    gkf = GroupKFold(n_splits=5)
    results = []
    
    print("\nStarting cross-validation...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(author_training, author_training['appeared_target'], groups)):
        print(f"\nFold {fold_idx + 1}/5")
        
        X_train = author_training.iloc[train_idx][base_features].fillna(0)
        y_train = author_training.iloc[train_idx]['appeared_target']
        X_test = author_training.iloc[test_idx][base_features].fillna(0)
        y_test = author_training.iloc[test_idx]['appeared_target']
        
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
        
        # Train models
        gb = GradientBoostingClassifier(random_state=42, max_depth=3, min_samples_split=20, min_samples_leaf=10)
        lr = LogisticRegression(max_iter=5000, random_state=42, penalty='l2')
        
        gb_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.05]}
        lr_grid = {'C': [0.01, 0.1, 1]}
        
        gb_cv = GridSearchCV(gb, gb_grid, cv=3, n_jobs=-1, scoring='roc_auc')
        lr_cv = GridSearchCV(lr, lr_grid, cv=3, n_jobs=-1, scoring='roc_auc')
        
        gb_cv.fit(X_train_selected, y_train)
        lr_cv.fit(X_train_selected, y_train)
        
        ensemble = VotingClassifier(estimators=[
            ('gb', gb_cv.best_estimator_),
            ('lr', lr_cv.best_estimator_)
        ], voting='soft')
        
        ensemble.fit(X_train_selected, y_train)
        calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
        calibrated.fit(X_train_selected, y_train)
        
        y_pred = calibrated.predict(X_test_selected)
        if hasattr(calibrated, 'predict_proba') and len(calibrated.classes_) == 2:
            y_proba = calibrated.predict_proba(X_test_selected)[:, 1]
        else:
            y_proba = np.full_like(y_test, fill_value=0.5, dtype=float)
        
        # Calculate metrics
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_proba)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            print(f"AUC: {auc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
        else:
            auc = prec = rec = f1 = np.nan
            print("Only one class present in test set")
        
        results.append({'auc': auc, 'precision': prec, 'recall': rec, 'f1': f1})
    
    # Print overall results
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
    
    # Train final model on training data only
    print("\nTraining final model...")
    X_train_final = author_training[base_features].fillna(0)
    y_train_final = author_training['appeared_target']
    
    scaler_final = StandardScaler()
    X_train_scaled_final = scaler_final.fit_transform(X_train_final)
    X_train_scaled_final = clean_features(X_train_scaled_final)
    
    selector_final = SelectKBest(f_classif, k=min(10, len(base_features)))
    X_train_selected_final = selector_final.fit_transform(X_train_scaled_final, y_train_final)
    
    # Final ensemble
    gb_final = GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.05, 
                                         max_depth=3, min_samples_split=20, min_samples_leaf=10)
    lr_final = LogisticRegression(max_iter=5000, random_state=42, C=0.1, penalty='l2')
    
    gb_final.fit(X_train_selected_final, y_train_final)
    lr_final.fit(X_train_selected_final, y_train_final)
    
    ensemble_final = VotingClassifier(estimators=[
        ('gb', gb_final),
        ('lr', lr_final)
    ], voting='soft')
    ensemble_final.fit(X_train_selected_final, y_train_final)
    
    # Generate predictions for ALL authors (but with different logic for new vs experienced authors)
    X_all = author_full[base_features].fillna(0)
    X_all_scaled = scaler_final.transform(X_all)
    X_all_scaled = clean_features(X_all_scaled)
    X_all_selected = selector_final.transform(X_all_scaled)
    
    # Get base probabilities from the model
    probs_base = ensemble_final.predict_proba(X_all_selected)[:, 1]
    
    # Apply logic-based adjustments
    probs_adjusted = probs_base.copy()
    
    # For authors with 0 participations, cap probability at 10%
    zero_participation_mask = author_full['num_participations'] == 0
    probs_adjusted[zero_participation_mask] = np.minimum(probs_adjusted[zero_participation_mask], 0.10)
    
    # For authors with 1 participation, cap probability at 30%
    one_participation_mask = author_full['num_participations'] == 1
    probs_adjusted[one_participation_mask] = np.minimum(probs_adjusted[one_participation_mask], 0.30)
    
    # Boost probability for authors with recent participation (within last 2 years)
    recent_mask = author_full['years_since_last'] <= 2
    probs_adjusted[recent_mask] = probs_adjusted[recent_mask] * 1.2
    probs_adjusted = np.minimum(probs_adjusted, 1.0)  # Cap at 1.0
    
    # Apply conservative threshold (now on adjusted probabilities)
    conservative_threshold = np.percentile(probs_adjusted[training_mask], 85)  # Use training authors for threshold
    pred_conservative = (probs_adjusted >= conservative_threshold).astype(int)
    
    # Create results DataFrame
    author_full['probability'] = probs_adjusted
    author_full['prediction'] = pred_conservative
    author_full['confidence_percent'] = 100.0 * probs_adjusted
    
    # Save results
    results_to_save = author_full[['Author', 'prediction', 'probability', 'confidence_percent', 
                                  'num_participations', 'years_since_last', 'participation_rate']].copy()
    results_to_save = results_to_save.sort_values('probability', ascending=False)
    results_to_save['rank'] = range(1, len(results_to_save) + 1)
    
    # Rename for clarity
    results_to_save = results_to_save.rename(columns={
        'Author': 'predicted_author',
        'prediction': 'will_participate_2026',
        'probability': 'participation_probability'
    })
    
    output_file = '../../data/predictions/neurips_2026_predictions.csv'
    results_to_save.to_csv(output_file, index=False)
    
    print(f"\n=== NeurIPS 2026 Predictions Complete ===")
    print(f"Predictions saved to: {output_file}")
    print(f"Total authors analyzed: {len(author_full)}")
    print(f"Authors predicted to participate: {results_to_save['will_participate_2026'].sum()}")
    print(f"Conservative threshold: {conservative_threshold:.3f}")
    
    print(f"\nTop 10 most likely to participate:")
    top_10 = results_to_save.head(10)[['predicted_author', 'participation_probability', 'confidence_percent', 'num_participations']]
    for _, row in top_10.iterrows():
        print(f"{row['predicted_author']}: {row['participation_probability']:.3f} ({row['confidence_percent']:.1f}% confidence, {row['num_participations']} past participations)")
    
    # Save the model
    joblib.dump(ensemble_final, '../../data/processed/neurips_participation_model.pkl')
    print(f"\nModel saved as: ../../data/processed/neurips_participation_model.pkl")

if __name__ == "__main__":
    main()
