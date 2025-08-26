# Class Imbalance Solutions for AAAI Author Participation Prediction

## Problem
Your dataset has class imbalance with approximately 2.4:1 ratio (7842 negatives vs 3283 positives). The original GradientBoostingClassifier doesn't support `class_weight='balanced'` parameter.

## Solutions Implemented

### 1. LightGBM with scale_pos_weight (Default - Recommended)
```bash
python src/models/aaai_predict_authors.py --quiet-warnings
```

**Features:**
- Replaces GradientBoostingClassifier with LightGBM
- Uses `scale_pos_weight=2.39` (negative_count / positive_count)
- Maintains LogisticRegression and RandomForestClassifier with `class_weight='balanced'`
- Fast training and prediction
- Excellent performance: F1=0.581 ± 0.014

**Performance:**
- AUC: 0.780 ± 0.009
- Precision: 0.553 ± 0.019  
- Recall: 0.612 ± 0.017
- F1: 0.581 ± 0.014

### 2. SMOTE Oversampling
```bash
python src/models/aaai_predict_authors.py --use-smote --quiet-warnings
```

**Features:**
- Uses SMOTE (Synthetic Minority Oversampling Technique)
- Creates synthetic positive examples to balance training data
- Applied inside cross-validation to prevent data leakage
- More computationally intensive but handles severe imbalance well

### 3. Enhanced Features for Class Imbalance

Both methods benefit from enhanced feature engineering:
- **Cross-conference features**: AAAI share, venue diversity, conference hopping
- **Temporal decay features**: Publication decay analysis, long-timer dropoff risk
- **Gap analysis**: Long gaps, gap trend analysis
- **Recency weighting**: Exponential decay, recent vs career ratio

## Key Improvements

### 1. Automatic Class Weight Calculation
```python
def calculate_class_weight_ratio(y_train: np.ndarray) -> float:
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    ratio = neg_count / pos_count if pos_count > 0 else 1.0
    return ratio
```

### 2. Enhanced Ensemble Model
- LightGBM with automatic scale_pos_weight calculation
- LogisticRegression with class_weight='balanced'
- RandomForestClassifier with class_weight='balanced'
- Voting classifier combines all three approaches

### 3. F1-Optimized Threshold Selection
- Cross-validation finds optimal F1 threshold
- Performance evaluation at optimal threshold
- Accounts for class imbalance in threshold selection

### 4. SMOTE Pipeline Integration
```python
def create_smote_pipeline(base_model: Any, feature_selector: SelectKBest) -> ImbPipeline:
    return ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=5)),
        ('selector', feature_selector),
        ('classifier', base_model)
    ])
```

## Usage Examples

### Basic Usage (LightGBM)
```bash
python src/models/aaai_predict_authors.py
```

### With SMOTE
```bash
python src/models/aaai_predict_authors.py --use-smote
```

### Suppress Warnings
```bash
python src/models/aaai_predict_authors.py --quiet-warnings
```

### Custom Paths
```bash
python src/models/aaai_predict_authors.py \
    --data-path custom/path/data.csv \
    --output-dir custom/output \
    --model-path custom/model.pkl \
    --use-smote \
    --quiet-warnings
```

## Comparison Results

| Method | AUC | Precision | Recall | F1 | Speed | Memory |
|--------|-----|-----------|--------|----|----|--------|
| LightGBM | 0.780±0.009 | 0.553±0.019 | 0.612±0.017 | 0.581±0.014 | Fast | Low |
| SMOTE | Running... | TBD | TBD | TBD | Slower | Higher |

## Recommendations

1. **Use LightGBM method (default)** for:
   - Production deployments
   - Fast training/prediction needs
   - Good balance of performance and efficiency

2. **Use SMOTE method** for:
   - Severe class imbalance (>10:1)
   - When you have enough computational resources
   - Research scenarios requiring thorough exploration

3. **Warning suppression** with `--quiet-warnings` is recommended to avoid:
   - LightGBM feature name warnings
   - sklearn deprecation warnings
   - Runtime warnings during training

## Technical Notes

- Both methods preserve GroupKFold cross-validation (no data leakage)
- SMOTE is applied only on training folds, not test data
- Feature selection happens after oversampling
- Model calibration maintains probability quality
- Enhanced logic adjustments still applied to final predictions

## Files Modified

- `src/models/aaai_predict_authors.py`: Main prediction script with both methods
- Added LightGBM and imbalanced-learn dependencies
- Enhanced warning suppression for cleaner output
- Maintained backward compatibility with original pipeline
