import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error, roc_auc_score
from collections import defaultdict, Counter
import re
from sklearn.preprocessing import StandardScaler


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

all_years = sorted(df['year'].unique())
min_year, max_year = min(all_years), max(all_years)
base_features = ['num_participations', 'last_year', 'first_year', 'max_consecutive_years', 'years_since_last', 'years_since_first', 'exp_decay_sum', 'markov_prob']

def get_year_features(years, all_years):
    years_set = set(years)
    # Participation in each year
    year_feats = {f'appeared_{y}': int(y in years_set) for y in all_years}
    # Recent streak: how many consecutive years up to the last year
    sorted_years = sorted(years)
    streak = 1
    max_streak = 1
    for i in range(1, len(sorted_years)):
        if sorted_years[i] == sorted_years[i-1] + 1:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 1
    # Years since first/last participation
    last_year = sorted_years[-1]
    first_year = sorted_years[0]
    years_since_last = max_year - last_year
    years_since_first = max_year - first_year
    # Markov: transition probabilities (simple)
    transitions = Counter()
    for i in range(1, len(sorted_years)):
        transitions[(sorted_years[i-1], sorted_years[i])] += 1
    # Temporal recency: exponential decay sum
    decay = 0.5  # You can tune this
    exp_decay_sum = sum(np.exp(-decay * (max(all_years) - y)) for y in years)
    # Markov: probability of participating given previous year
    markov_prob = 0.0
    if len(years) > 1:
        transitions_count = 0
        for i in range(1, len(years)):
            if years[i] == years[i-1] + 1:
                transitions_count += 1
        markov_prob = transitions_count / (len(years) - 1)
    return {
        'num_participations': len(years),
        'last_year': last_year,
        'first_year': first_year,
        'max_consecutive_years': max_streak,
        'years_since_last': years_since_last,
        'years_since_first': years_since_first,
        'exp_decay_sum': exp_decay_sum,
        'markov_prob': markov_prob,
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
results = []
confidence_lookup = {}
for test_year in range(min_year+3, max_year+1):
    train_years = [y for y in all_years if y < test_year]
    feature_cols = [f'appeared_{y}' for y in train_years]
    features = base_features + feature_cols
    X_train = author_full[features].fillna(0)
    y_train = author_full.get(f'appeared_{test_year}', 0)
    X_test = X_train.copy()
    y_test = y_train if isinstance(y_train, pd.Series) else pd.Series([y_train]*len(X_test), index=X_test.index)

    if not isinstance(y_train, (pd.Series, np.ndarray)) or len(np.unique(y_train)) < 2:
        print(f"[Skipping year {test_year}] Not enough class diversity in training labels.")
        continue

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(max_iter=5000, random_state=42)

    rf_grid = {'n_estimators': [100, 200], 'max_depth': [None, 5]}
    gb_grid = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}
    lr_grid = {'C': [0.1, 1, 10]}

    rf_cv = GridSearchCV(rf, rf_grid, cv=3, n_jobs=-1)
    gb_cv = GridSearchCV(gb, gb_grid, cv=3, n_jobs=-1)
    lr_cv = GridSearchCV(lr, lr_grid, cv=3, n_jobs=-1)
    rf_cv.fit(X_train_scaled, y_train)
    gb_cv.fit(X_train_scaled, y_train)
    lr_cv.fit(X_train_scaled, y_train)

    ensemble = VotingClassifier(estimators=[
        ('rf', rf_cv.best_estimator_),
        ('gb', gb_cv.best_estimator_),
        ('lr', lr_cv.best_estimator_)
    ], voting='soft')
    ensemble.fit(X_train_scaled, y_train)
    y_pred = ensemble.predict(X_test_scaled)
    if hasattr(ensemble, 'predict_proba') and len(ensemble.classes_) == 2:
        y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    else:
        single_class = ensemble.classes_[0]
        y_proba = np.full_like(y_test, fill_value=single_class, dtype=float)
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = np.nan
        print(f"[Warning] Only one class present in y_test for year {test_year}. ROC AUC is undefined.")
    report = classification_report(y_test, y_pred, output_dict=True)
    results.append({'test_year': test_year, 'auc': auc, 'report': report})
    print(f"Year {test_year} AUC: {auc if not np.isnan(auc) else 'undefined (one class)'}")
    print(classification_report(y_test, y_pred))

    # --- Track confidence by participation pattern ---
    import collections.abc
    if isinstance(y_pred, (np.ndarray, list, collections.abc.Sequence)) and len(y_pred) == len(y_test):
        for i in range(len(y_pred)):
            row = author_full.iloc[i]
            pattern = (row['num_participations'], row['max_consecutive_years'])
            pred = y_pred[i]
            true = y_test.iloc[i]
            if pattern not in confidence_lookup:
                confidence_lookup[pattern] = {'correct': 0, 'total': 0}
            if pred == true:
                confidence_lookup[pattern]['correct'] += 1
            confidence_lookup[pattern]['total'] += 1

# Use ensemble for final prediction (train on all years up to max_year)
predict_year = max_year + 1
feature_years_pred = [y for y in all_years if y < predict_year]
feature_cols_pred = [f'appeared_{y}' for y in feature_years_pred]
features_pred = base_features + feature_cols_pred
X_pred = author_full[features_pred].fillna(0)

# Scale features for final prediction
scaler_final = StandardScaler()
X_pred_scaled = scaler_final.fit_transform(X_pred)

rf_final = RandomForestClassifier(random_state=42, n_estimators=rf_cv.best_params_['n_estimators'], max_depth=rf_cv.best_params_['max_depth'])
gb_final = GradientBoostingClassifier(random_state=42, n_estimators=gb_cv.best_params_['n_estimators'], learning_rate=gb_cv.best_params_['learning_rate'])
lr_final = LogisticRegression(max_iter=5000, random_state=42, C=lr_cv.best_params_['C'])

ensemble_final = VotingClassifier(estimators=[
    ('rf', rf_final),
    ('gb', gb_final),
    ('lr', lr_final)
], voting='soft')
ensemble_final.fit(X_pred_scaled, author_full.get(f'appeared_{max_year}', 0))

author_full['predict_next'] = ensemble_final.predict(X_pred_scaled)
if hasattr(ensemble_final, 'predict_proba') and len(ensemble_final.classes_) == 2:
    author_full['predict_next_proba'] = ensemble_final.predict_proba(X_pred_scaled)[:, 1]
else:
    single_class = ensemble_final.classes_[0]
    author_full['predict_next_proba'] = np.full(X_pred.shape[0], fill_value=single_class, dtype=float)
author_full['predict_2026'] = ensemble_final.predict(X_pred_scaled)

# --- Assign confidence percentage based on validation splits ---
def get_confidence(row):
    pattern = (row['num_participations'], row['max_consecutive_years'])
    stats = confidence_lookup.get(pattern, None)
    if stats and stats['total'] > 0:
        return 100.0 * stats['correct'] / stats['total']
    else:
        return np.nan
author_full['confidence_percent'] = author_full.apply(get_confidence, axis=1)


# Paper count regression: predict number of papers for next year
author_year_paper_counts = df.groupby(['author', 'year']).size().reset_index(name='papers_per_year')
author_paper_counts = df.groupby('author').size().reset_index(name='num_papers_submitted')

# Prepare regression data
regression_rows = []
for _, row in author_year_paper_counts.iterrows():
    author = row['author']
    year = row['year']
    # Use features up to year-1
    feats = author_full[author_full['author'] == author]
    if feats.empty: continue
    feats = feats.iloc[0]
    regression_rows.append({
        'author': author,
        'year': year,
        'papers_per_year': row['papers_per_year'],
        **{k: feats[k] for k in base_features if k in feats}
    })
reg_df = pd.DataFrame(regression_rows)


reg_train = reg_df[reg_df['year'] < max_year]
reg_test = reg_df[reg_df['year'] == max_year]
reg_features = base_features
reg_X_train = reg_train[reg_features]
reg_y_train = reg_train['papers_per_year']
reg_X_test = reg_test[reg_features]
reg_y_test = reg_test['papers_per_year'] if not reg_test.empty else None

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(reg_X_train, reg_y_train)
if not reg_X_test.empty and reg_y_test is not None:
    reg_pred = regressor.predict(reg_X_test)
    print("Paper count regression RMSE:", np.sqrt(mean_squared_error(reg_y_test, reg_pred)))

# Predict for next year for all authors
author_full['expected_papers_next'] = regressor.predict(author_full[reg_features].fillna(0))
author_full['expected_papers_next'] = author_full['expected_papers_next'].apply(lambda x: max(0, round(x)))


# Prepare final output
# Prepare final output
final_df = author_full.copy()
final_df = final_df.merge(author_paper_counts[['author', 'num_papers_submitted']], on='author', how='left')
final_df['predicted_next_participation_year'] = predict_year * final_df['predict_next']
final_df.loc[final_df['predicted_next_participation_year'] == 0, 'predicted_next_participation_year'] = ''


final_df_out = final_df[['author',
                        'predicted_next_participation_year',
                        'num_participations',
                        'first_year',
                        'last_year',
                        'num_papers_submitted',
                        'expected_papers_next',
                        'predict_next_proba',
                        'confidence_percent']]
final_df_out = final_df_out.rename(columns={
    'author': 'Author',
    'predicted_next_participation_year': 'Predicted next participation year',
    'num_participations': 'Number of times participated',
    'first_year': 'First participated year',
    'last_year': 'Last participated year',
    'num_papers_submitted': 'Number of papers submitted',
    'expected_papers_next': 'Number of papers expected to submit in upcoming year',
    'predict_next_proba': 'Probability of participation in upcoming year',
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

