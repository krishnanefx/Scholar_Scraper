from sklearn.calibration import CalibratedClassifierCV
def clean_features(X):
    """Replace NaN, inf, -inf with 0 and clip extreme values."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Optionally, clip to a reasonable range to avoid overflows
    X = np.clip(X, -1e6, 1e6)
    return X
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error, roc_auc_score, precision_score, recall_score, f1_score, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from collections import defaultdict, Counter
import re
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve


# Load the data
df = pd.read_csv('aaai25_papers_authors_split.csv')

# Data Quality: Clean and normalize author names
def normalize_author(name):
    if pd.isnull(name):
        return ''
    name = name.strip()
    name = re.sub(r'\s+', ' ', name)
    name = name.lower()
    name = name.replace('.', '')
    name = re.sub(r'[^a-z\s]', '', name)
    return name.title()

df['author'] = df['author'].fillna('').apply(normalize_author)
df = df[df['author'] != '']

# Handle missing/inconsistent years and paper counts
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)



# Feature engineering: richer author participation history

all_years_full = sorted(df['year'].unique())
min_year, max_year = min(all_years_full), max(all_years_full)
target_year = max_year
# For feature engineering, exclude the target year (t) so features only use history up to t-1
all_years = [y for y in all_years_full if y < target_year]
base_features = [
    'num_participations',
    'max_consecutive_years',
    'years_since_last',
    'years_since_first',
    'exp_decay_sum',
    'markov_prob',
    'participation_rate',
    'max_gap',
    'mean_gap',
    'streak_length'
]

def get_year_features(years, all_years):
    years_set = set(years)
    # Participation in each year (lagged: only up to t-1)
    year_feats = {f'appeared_{y}': int(y in years_set) for y in all_years}
    # Only use years < max(all_years) for lagged features
    past_years_attended = sorted([y for y in years if y in all_years])
    # Streak features (lagged)
    streak = 1
    max_streak = 1
    streak_length = 1
    for i in range(1, len(past_years_attended)):
        if past_years_attended[i] == past_years_attended[i-1] + 1:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 1
    streak_length = max_streak
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
    # Markov: transition probabilities (simple, lagged)
    transitions = Counter()
    for i in range(1, len(past_years_attended)):
        transitions[(past_years_attended[i-1], past_years_attended[i])] += 1
    # Temporal recency: exponential decay sum (lagged)
    decay = 0.5  # You can tune this
    exp_decay_sum = sum(np.exp(-decay * (max(all_years) - y)) for y in past_years_attended)
    # Markov: probability of participating given previous year (lagged)
    markov_prob = 0.0
    if len(past_years_attended) > 1:
        transitions_count = 0
        for i in range(1, len(past_years_attended)):
            if past_years_attended[i] == past_years_attended[i-1] + 1:
                transitions_count += 1
        markov_prob = transitions_count / (len(past_years_attended) - 1)
    career_length = last_year - first_year + 1 if last_year >= first_year else 1
    normalized_participation_rate = len(past_years_attended) / career_length if career_length > 0 else 0
    return {
        'num_participations': len(past_years_attended),
        'last_year': last_year,
        'first_year': first_year,
        'career_length': career_length,
        'normalized_participation_rate': normalized_participation_rate,
        'max_consecutive_years': max_streak,
        'years_since_last': years_since_last,
        'years_since_first': years_since_first,
        'exp_decay_sum': exp_decay_sum,
        'markov_prob': markov_prob,
        'participation_rate': participation_rate,
        'max_gap': max_gap,
        'mean_gap': mean_gap,
        'streak_length': streak_length,
        **year_feats,
        'markov_transitions': dict(transitions)
    }

author_years = df.groupby('author')['year'].apply(list).reset_index()
author_years_feats = author_years['year'].apply(lambda years: get_year_features(years, all_years))
author_feats_df = pd.DataFrame(list(author_years_feats))
author_full = pd.concat([author_years['author'], author_feats_df], axis=1)

# Time-based validation: rolling window

# Time-based cross-validation: train on all years before, test on the next year

# --- Validation split with confidence tracking ---

# --- Per-author cross-validation for model evaluation ---
from sklearn.model_selection import GroupKFold
author_val_stats = {}  # author: {'correct': int, 'total': int, 'probas': [], 'truths': []}
results = []
confidence_lookup = {}

groups = author_full['author']
gkf = GroupKFold(n_splits=5)
author_val_stats = {}  # author: {'correct': int, 'total': int, 'probas': [], 'truths': []}
results = []
confidence_lookup = {}
target_year = max_year

# Before the GroupKFold split, create the target variable as a Series
author_full['appeared_target'] = author_full['author'].isin(
    df[df['year'] == target_year]['author']
).astype(int)
# Then use this for y in the split:
fold_preds, fold_probas, fold_truths = [], [], []

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(author_full, author_full['appeared_target'], groups)):
    # Lag all yearly indicators: when predicting year t, only include features from years ≤ t-1
    # Drop all binary year indicators for this experiment
    features = base_features
    X_train = author_full.iloc[train_idx][features].fillna(0)
    y_train = author_full.iloc[train_idx]['appeared_target']
    X_test = author_full.iloc[test_idx][features].fillna(0)
    y_test = author_full.iloc[test_idx]['appeared_target']
    # Print target distribution for each fold
    print(f"Train target distribution: {np.bincount(y_train.astype(int)) if len(y_train) > 0 else 'empty'}")
    print(f"Test target distribution: {np.bincount(y_test.astype(int)) if len(y_test) > 0 else 'empty'}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = clean_features(X_train_scaled)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = clean_features(X_test_scaled)

    # Use only probabilistic models for probability estimates
    gb = GradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(max_iter=5000, random_state=42)
    gb_grid = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}
    lr_grid = {'C': [0.1, 1, 10]}
    gb_cv = GridSearchCV(gb, gb_grid, cv=3, n_jobs=-1)
    lr_cv = GridSearchCV(lr, lr_grid, cv=3, n_jobs=-1)
    gb_cv.fit(X_train_scaled, y_train)
    lr_cv.fit(X_train_scaled, y_train)

    ensemble = VotingClassifier(estimators=[
        ('gb', gb_cv.best_estimator_),
        ('lr', lr_cv.best_estimator_)
    ], voting='soft')

    # Fit ensemble first, then calibrate using isotonic regression
    ensemble.fit(X_train_scaled, y_train)
    calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv='prefit')
    calibrated.fit(X_train_scaled, y_train)
    y_pred = calibrated.predict(X_test_scaled)
    if hasattr(calibrated, 'predict_proba') and len(calibrated.classes_) == 2:
        y_proba = calibrated.predict_proba(X_test_scaled)[:, 1]
    else:
        single_class = calibrated.classes_[0]
        y_proba = np.full_like(y_test, fill_value=single_class, dtype=float)

    # --- Metrics ---
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"AUC: {auc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
    else:
        auc = np.nan
        prec = rec = f1 = np.nan
        print(f"[Warning] Only one class present in y_test for group split. ROC AUC is undefined.")
    report = classification_report(y_test, y_pred, output_dict=True)
    results.append({'auc': auc, 'precision': prec, 'recall': rec, 'f1': f1, 'report': report})
    print(f"Group split AUC: {auc if not np.isnan(auc) else 'undefined (one class)'}")
    print(classification_report(y_test, y_pred))

    # --- Calibration: reliability plot ---
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    plt.figure(figsize=(4, 4))
    plt.plot(prob_pred, prob_true, marker='o', label='Fold')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Reliability (Calibration) Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Store per-fold predictions and probabilities ---
    fold_preds.extend(y_pred)
    fold_probas.extend(y_proba)
    fold_truths.extend(y_test)

# Use ensemble for final prediction (train on all years up to max_year)
predict_year = max_year + 1
feature_years_pred = [y for y in all_years if y < predict_year]
feature_cols_pred = [f'appeared_{y}' for y in feature_years_pred]
features_pred = base_features + feature_cols_pred
X_pred = author_full[features_pred].fillna(0)


# Scale features for final prediction
scaler_final = StandardScaler()
X_pred_scaled = scaler_final.fit_transform(X_pred)
X_pred_scaled = clean_features(X_pred_scaled)


# Use only probabilistic models for final prediction ensemble
gb_final = GradientBoostingClassifier(random_state=42, n_estimators=200, learning_rate=0.1)
lr_final = LogisticRegression(max_iter=5000, random_state=42, C=1)
ensemble_final = VotingClassifier(estimators=[
    ('gb', gb_final),
    ('lr', lr_final)
], voting='soft')
ensemble_final.fit(X_pred_scaled, author_full['appeared_target'])

author_full['predict_next'] = ensemble_final.predict(X_pred_scaled)

author_full['predict_2026'] = ensemble_final.predict(X_pred_scaled)

# --- Assign confidence percentage based on validation splits and participation history ---
def get_confidence(row):
    author = row['author']
    stats = author_val_stats.get(author, None)
    # 1. Per-author validation accuracy
    if stats and stats['total'] > 0:
        acc_conf = stats['correct'] / stats['total']
    else:
        acc_conf = np.nan
    # 2. Model calibration: mean calibrated probability for true class
    if stats and stats['probas'] and stats['truths']:
        cal_conf = np.mean([p if t == 1 else 1-p for p, t in zip(stats['probas'], stats['truths'])])
    else:
        cal_conf = np.nan
    # 3. Temper by participation history length
    n_part = row.get('num_participations', 0)
    temper = min(1.0, 0.5 + 0.5 * np.tanh((n_part-2)/3))  # 0.5 for very sparse, ~1 for moderate+ history
    if not np.isnan(cal_conf):
        return 100.0 * cal_conf * temper
    elif not np.isnan(acc_conf):
        return 100.0 * acc_conf * temper
    else:
        return np.nan

author_full['confidence_percent'] = author_full.apply(get_confidence, axis=1)

# --- Paper count: bootstrapped prediction intervals ---
def bootstrap_paper_count(model, X, n_boot=100):
    preds = []
    n = X.shape[0]
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        Xb = X.iloc[idx]
        preds.append(model.predict(Xb))
    preds = np.array(preds)
    mean_pred = np.mean(preds, axis=0)
    lower = np.percentile(preds, 2.5, axis=0)
    upper = np.percentile(preds, 97.5, axis=0)
    return mean_pred, lower, upper

mean_rf, lower_rf, upper_rf = bootstrap_paper_count(rf_reg, reg_X_pred_clean)
mean_hgb, lower_hgb, upper_hgb = bootstrap_paper_count(hgb_reg, reg_X_pred_clean)
mean_ens = 0.5 * mean_rf + 0.5 * mean_hgb
lower_ens = 0.5 * lower_rf + 0.5 * lower_hgb
upper_ens = 0.5 * upper_rf + 0.5 * upper_hgb

author_full['expected_papers_next'] = mean_ens
author_full['expected_papers_next'] = author_full['expected_papers_next'].apply(lambda x: max(0, round(x)))
author_full['expected_papers_next_lower'] = lower_ens
author_full['expected_papers_next_upper'] = upper_ens


# After computing bootstrapped intervals for paper count
print('Sample expected paper count predictions (mean ± 95% CI):')
for i, row in author_full.head(10).iterrows():
    print(f"{row['author']}: {row['expected_papers_next']} (95% CI: {row['expected_papers_next_lower']:.2f}–{row['expected_papers_next_upper']:.2f})")

# Print sample participation confidence scores
print('\nSample participation confidence scores:')
for i, row in author_full.head(10).iterrows():
    print(f"{row['author']}: {row['confidence_percent']:.1f}%")

# Prepare final output
# Prepare final output
final_df = author_full.copy()
final_df = final_df.merge(author_paper_counts[['author', 'num_papers_submitted']], on='author', how='left')
final_df['predicted_next_participation_year'] = predict_year * final_df['predict_next']
# Cast to object type to allow string assignment and avoid FutureWarning
final_df['predicted_next_participation_year'] = final_df['predicted_next_participation_year'].astype(object)
final_df.loc[final_df['predicted_next_participation_year'] == 0, 'predicted_next_participation_year'] = ''


final_df_out = final_df[['author',
                        'predicted_next_participation_year',
                        'num_participations',
                        'first_year',
                        'last_year',
                        'num_papers_submitted',
                        'expected_papers_next',
                        'confidence_percent']]
final_df_out = final_df_out.rename(columns={
    'author': 'Author',
    'predicted_next_participation_year': 'Predicted next participation year',
    'num_participations': 'Number of times participated',
    'first_year': 'First participated year',
    'last_year': 'Last participated year',
    'num_papers_submitted': 'Number of papers submitted',
    'expected_papers_next': 'Number of papers expected to submit in upcoming year',
    'confidence_percent': 'Confidence in prediction (%)'
})
final_df_out.to_csv('author_participation_predictions.csv', index=False)
print('Final author participation predictions saved to author_participation_predictions.csv (with confidence percentages)')


# Bar graph: number of unique authors per year (actual only, no 2026 prediction)
authors_per_year = df.groupby('year')['author'].nunique()
years = [int(y) for y in authors_per_year.index]
counts = [int(c) for c in authors_per_year.values]
plt.figure(figsize=(12,6))
bars = plt.bar(years, counts, color='skyblue')
plt.xlabel('Year')
plt.ylabel('Number of Unique Authors')
plt.title('Number of Unique Authors Submitted per Year')
plt.tight_layout()
plt.xticks(ticks=years, labels=[str(y) for y in years], rotation=45)
plt.savefig('authors_per_year_bargraph.png')
plt.close()
print('Bar graph of number of unique authors per year saved to authors_per_year_bargraph.png')

