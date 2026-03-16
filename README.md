README

## Model Development Summary

### **Objective**
Build a regression model to predict `avg_price` using geographic location (lat, lng), building characteristics (m2, build_year, build_month, building_age, households), transaction data (avg_sales), and apartment score.

### **Model Selection Journey**

**Ridge Regression (Baseline)**
- **Why over Linear Regression?** L2 regularization prevents overfitting
- **Why over Lasso?** Preserves all features for interpretability  
- **Why over Decision Tree?** Stable, mathematically elegant baseline
- **Purpose:** Establish performance floor, understand linear relationships while preserving features

**XGBoost (Primary Model)**
- **Why over Random Forest?** Superior performance on tabular data
- **Why over Neural Networks?** Works better with smaller dataset (~3.9k samples)
- **Why over LightGBM?** Better documentation for reproducibility
- **Purpose:** Capture complex non-linear patterns and interactions

**Phase 1: Modeling**
- Started with Ridge (linear) and XGBoost (tree-based) models

**Phase 2: Transformation**
- Applied `log1p` transformation to target for ALL models
- Used `TransformedTargetRegressor` wrapper for consistent transformation
- Result: XGBoost performance skyrocketed to R² = 0.77

**Phase 3: Performance Comparison**
| Model | Test R² | Overfitting Gap | Key Insight |
|-------|---------|----------------|-------------|
| Ridge | 0.33 | 0.11 | Simple but limited |
| XGBoost | 0.77 | 0.12 | **Best performer** |
| Tuned XGBoost | 0.77 | 0.19 | Tuning hurt generalization |

**Phase 4: Overfitting Control**
- Dataset: 3,144 training samples, 787 test samples
- Applied regularization: increase n_estimators, lower learning rate, shallower trees, feature sampling, increase reg_lambda to make model more “conservative”
- **Final overfitting gap: 0.12** 

### **Final Model Selection**
**Chosen Model**: XGBoost with log-transformation and regularization
- **Test R²**: 0.7692 (explains 77% of variance)
- **Average Error**: 16.5% (MAPE)
- **Key Parameters**:
  - `learning_rate`: 0.05
  - `max_depth`: 5
  - `n_estimators`: 200
  - Regularization: L1/L2, feature sampling, bagging

### **Key Technical Decisions**
1. **Transformation**: `log1p` for handling skewed targets and zeros
2. **Validation**: 80/20 train-test split with 787 test samples
3. **Regularization**: Prioritized generalization over training accuracy
4. **Metric Focus**: Test set performance over training metrics

### **Lessons Learned**
1. **Consistency is key**: Same preprocessing for all models
2. **Test set matters**: Always evaluate on unseen data
3. **Overfitting control**: More important than perfect training fit
4. **Simplicity**: Default XGBoost often outperforms aggressively tuned versions