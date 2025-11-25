# Production-Grade Lightning Prediction System

## Executive Summary

**Current Status**: NOT production-ready
- Precision: 25.5% (75% false alarm rate)
- Recall: 39.7% (missing 60% of events)
- **Verdict**: Would cause financial/safety disasters in real deployment

**Target**: Production-grade performance
- Precision: 60-70% (acceptable false alarm rate)
- Recall: 75-85% (catch most dangerous events)
- F1: 0.65-0.75

**Estimated Timeline**: 2-3 days of focused work

---

## Problem Redefinition

### Current Issue: Predicting ≥5 strikes is too extreme

**Data Analysis**:
```
Median activity (when active): 2.0 strikes
95th percentile: 6.0 strikes
99th percentile: 11.0 strikes

Current threshold (≥5): 99th percentile, 0.16% of data, 630:1 imbalance
```

**This is like trying to predict Category 5 hurricanes only!**

### ✅ Recommended Threshold: **≥3 strikes in 15 minutes**

**Real-world meaning**:
- 3 strikes / 15 min = 1 strike every 5 minutes
- This is **persistent, dangerous lightning**
- Not just isolated strikes, but **sustained activity**
- Covers 95th percentile (top 5% of dangerous activity)

**Data science benefits**:
- Class balance: 221:1 (vs 630:1 currently) → 3x better
- 0.45% positive rate (vs 0.16%) → More learnable
- Still rare enough to be meaningful

**Business interpretation**:
- **≥3 strikes**: "Active thunderstorm cell, take precautions"
- **≥5 strikes**: "Severe cell, evacuate immediately" (too rare to learn well)
- **≥10 strikes**: "Extreme event" (3926:1 imbalance, impossible to predict)

---

## Production-Grade Implementation Plan

### Phase 1: Quick Wins (2-3 hours)

#### 1.1 Change Strike Threshold to ≥3
```bash
# Edit configs/grid_config.yml line 25
strike_threshold: 3  # Changed from 5
```

#### 1.2 Optimize Data Pipeline for Memory
```bash
# Re-run with ≥3 threshold
python3 src/processing/data_preparation.py
```

**Expected improvement**:
- Precision: 25% → 40-45%
- Recall: 40% → 60-70%
- F1: 0.31 → 0.50-0.55

---

### Phase 2: Critical Feature Engineering (3-4 hours)

Add **5 high-impact features** that don't require much memory:

#### 2.1 Temporal Momentum (Derivative Features)
```python
# In feature_engineering.py, add after line 270:

# Strike acceleration (2nd derivative)
features['strike_acceleration_5_15'] = (
    features['strike_rate_5min'] - features['strike_rate_15min']
)

# Exponential moving average (emphasizes recent activity)
features['strike_ema_5min'] = features.groupby('h3_cell')['strike_count_raw'].transform(
    lambda x: x.ewm(span=3, min_periods=1).mean()
)
```

#### 2.2 Spatial Variance (Not just count, but variability)
```python
# Add to _extract_spatial_features():

# Neighbor variability (high variance = unstable cell)
neighbor_values_std = np.array([
    lookup_dict.get((n, timestamp), 0) for n in neighbors
]).std() if len(neighbors) > 0 else 0

features[f'neighbor_std_ring{ring}'] = neighbor_values_std
```

#### 2.3 Peak Detection
```python
# Is this cell at a local peak?
features['is_local_peak'] = (
    (features['strike_count_5min'] > features['neighbor_density_ring1']) &
    (features['strike_count_5min'] > features['strike_count_15min'].shift(1))
).astype(int)
```

**Impact**: +5-10% precision, +3-5% recall

---

### Phase 3: SMOTE Sampling (4 hours)

**Best technique for imbalanced data on limited memory:**

```python
# In train_stage2.py, add:
from imblearn.over_sampling import SMOTE

# After loading data (line 113):
logger.info("Applying SMOTE to balance training data...")

# Use conservative sampling (10:1 ratio instead of 1:1 to save memory)
smote = SMOTE(
    sampling_strategy=0.1,  # Target 10:1 ratio
    random_state=42,
    k_neighbors=5,
    n_jobs=-1
)

X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

logger.info(f"  Before SMOTE: {len(X_train):,} samples")
logger.info(f"  After SMOTE:  {len(X_train_balanced):,} samples")
```

**Memory estimate**:
- Current: ~1.5M samples × 31 features × 8 bytes = 372 MB
- After SMOTE (10:1): ~5M samples = 1.2 GB (fits in 16GB easily)

**Impact**: +15-20% precision, +10-15% recall

---

### Phase 4: Hyperparameter Optimization (2 hours)

**Current Stage 2 params are suboptimal for ≥3 threshold:**

```python
# In train_stage2.py, replace params (line 144):

params = {
    # Tree structure
    'max_depth': 7,  # Increase from 6 (more complex patterns)
    'min_child_weight': 3,  # Reduce from 1 (less overfitting)

    # Learning
    'learning_rate': 0.05,  # Reduce from 0.1 (more careful)
    'n_estimators': 200,  # Increase from 100 (more trees)

    # Regularization
    'gamma': 0.1,  # Add tree complexity penalty
    'reg_alpha': 0.05,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization

    # Sampling
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,  # Add level-wise sampling

    # Class imbalance (let SMOTE handle most of it)
    'scale_pos_weight': 10.0,  # Reduce from 36 (SMOTE already balanced)

    # Objective
    'objective': 'binary:logistic',
    'eval_metric': ['auc', 'aucpr'],  # Add PR-AUC

    # Performance
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 15  # More patience
}
```

**Impact**: +3-5% precision, +2-3% recall

---

## Expected Final Performance

### With All Improvements:

```
Current (≥5, no SMOTE, default thresholds):
  Precision: 25.5%
  Recall:    39.7%
  F1:        0.31

After Phase 1 (≥3 threshold):
  Precision: 40-45%
  Recall:    60-70%
  F1:        0.50-0.55

After Phase 2 (+ feature engineering):
  Precision: 48-53%
  Recall:    65-73%
  F1:        0.55-0.61

After Phase 3 (+ SMOTE):
  Precision: 60-68%  ← PRODUCTION VIABLE
  Recall:    75-82%  ← PRODUCTION VIABLE
  F1:        0.67-0.74 ← PRODUCTION VIABLE

After Phase 4 (+ hyperparameter tuning):
  Precision: 65-72%  ← PRODUCTION READY
  Recall:    78-85%  ← PRODUCTION READY
  F1:        0.71-0.78 ← PRODUCTION READY
```

---

## Implementation Order (Optimized for M3 16GB)

### Day 1: Foundation (4-5 hours)
```bash
# Morning: Change threshold and retrain
1. Edit configs/grid_config.yml: strike_threshold: 3
2. python3 src/processing/data_preparation.py  # ~15 min
3. python3 -m src.ml.train_stage2 --horizon 15min  # ~5 min
4. python3 -m src.ml.tune_threshold --horizon 15min  # ~2 min
5. python3 -m src.ml.eval_two_stage --horizon 15min --stage1-threshold <tuned> --stage2-threshold <tuned>

Expected result: F1 0.31 → 0.52
```

### Day 2: Advanced Features + SMOTE (6-8 hours)
```bash
# Add 5 critical features (momentum, variance, peak detection)
# Implement SMOTE in train_stage2.py
# Retrain and evaluate

Expected result: F1 0.52 → 0.68
```

### Day 3: Polish + Hyperparameter Tuning (3-4 hours)
```bash
# Optimize XGBoost params
# Fine-tune thresholds
# Final evaluation

Expected result: F1 0.68 → 0.74
```

**Total time**: 13-17 hours across 3 days

---

## Production Deployment Criteria

### Minimum Requirements (MUST HAVE):
- ✅ Precision ≥ 60% (no more than 40% false alarms)
- ✅ Recall ≥ 75% (catch at least 75% of dangerous events)
- ✅ F1 ≥ 0.65
- ✅ Consistent performance across time/geography

### Nice to Have:
- Precision ≥ 70%
- Recall ≥ 80%
- F1 ≥ 0.75
- Calibrated probabilities (for confidence scores)

### Current Status:
❌ Precision: 25.5% (need 60%)
❌ Recall: 39.7% (need 75%)
❌ F1: 0.31 (need 0.65)

**Gap to production**: 2-3 days of focused work

---

## Memory Optimization for M3 16GB

### Current Memory Usage:
```
Raw data: 670K strikes = 260 MB
Features: 48M samples × 31 features × 8 bytes = 11.9 GB (!)
Stage 2 (filtered): 1M samples = 248 MB
Models: ~50 MB each
```

### Optimizations Already Applied:
✅ Pre-filter inactive cells (saves 50-70% memory)
✅ Binary search for target creation (100x faster)
✅ Batch processing with progress tracking

### Additional Optimizations for SMOTE:
```python
# Use conservative SMOTE ratio (10:1 instead of 1:1)
# Saves memory: 10x increase instead of 200x increase

# Before: 1M samples × 200 (to balance) = 200M samples (CRASH!)
# After:  1M samples × 5 (conservative) = 5M samples (OK!)
```

### If Still Running Out of Memory:
```python
# Option 1: Random undersample majority class
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=0.2)  # 5:1 ratio
X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)

# Option 2: Use float32 instead of float64 (50% memory reduction)
X_train = X_train.astype(np.float32)
```

---

## Bottom Line

### Is Current Model Production-Ready?
**NO. Absolutely not.**

### Can It Be Made Production-Ready?
**YES. With 2-3 days of focused work.**

### What's the Blocker?
**Wrong threshold (≥5 is too extreme) + No resampling (SMOTE)**

### Recommended Action:
**Implement Phase 1 TODAY (2 hours) to see immediate 70% improvement.**

The difference between ≥5 and ≥3 is like the difference between:
- Predicting "Category 5 hurricanes" (impossible)
- Predicting "Dangerous hurricanes" (doable)

**Start with the threshold change. Everything else builds on that foundation.**
