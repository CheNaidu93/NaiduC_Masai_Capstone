# NaiduC_Masai_Capstone
Masai Capstone project
# Analytics Pipeline — Zepto Capstone

## Overview

This module implements the Zepto Analytics Pipeline using the Titanic dataset.

The pipeline follows one continuous workflow:

```text
Titanic Dataset
      ↓
Profiling
      ↓
Cleaning
      ↓
Exploratory Data Analysis
      ↓
Visualization
      ↓
Train/Test Split
      ↓
Preprocessing
      ↓
Classification
      ↓
Model Evaluation
      ↓
Imbalance Analysis
      ↓
Hyperparameter Tuning
      ↓
Regression
      ↓
Model Recommendation
      ↓
Saved Complete Pipeline
```

The Titanic dataset is loaded through Seaborn's built-in loader and immediately saved as `titanic.csv` as the offline fallback. Subsequent processing uses the same saved/loaded dataset rather than independently downloading the dataset again.

---

# Installation

From the `analytics` directory:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running

Run:

```bash
python analytics_pipeline.py
```

The script generates:

* `titanic.csv`
* EDA charts
* correlation heatmap
* decision-tree visualization
* confusion matrices
* ROC curves
* regression residual plot
* classification comparison CSV
* imbalance comparison CSV
* regression metrics CSV
* complete fitted ML pipeline

The saved model is:

```text
artifacts/best_pipeline.joblib
```

---

# Dataset Profiling

The pipeline reports:

* DataFrame shape
* `df.info()`
* `df.describe()`
* missing-value percentages for every affected column

The raw Titanic dataset is saved using:

```python
df.to_csv("titanic.csv", index=False)
```

This CSV acts as the offline fallback for environments where the Seaborn dataset repository cannot be reached.

---

# Missing-Value Strategy

Missing values are handled according to the assignment threshold.

| Missing percentage | Strategy                                             |
| ------------------ | ---------------------------------------------------- |
| Less than 5%       | Drop affected rows                                   |
| 5%–30%             | Impute                                               |
| Greater than 30%   | Drop column or treat missing as an explicit category |

The exact measured missing percentages are printed by the pipeline at runtime.

For the Titanic dataset:

* `age` is imputed with its median because its missing percentage falls within the 5%–30% range.
* `embarked` has a small amount of missing data and affected rows are removed.
* `deck` has substantial missingness, so missing values are retained as an explicit `Unknown` category rather than attempting unreliable numerical-style imputation.
* `fare` is median-imputed if required.

The decisions are based on the measured percentages rather than arbitrary assumptions.

---

# Univariate Analysis

The pipeline produces:

```text
outputs/age_histogram.png
outputs/age_boxplot.png
outputs/fare_histogram.png
outputs/fare_boxplot.png
```

The IQR rule is used to identify outliers:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

The number of age and fare outliers is printed during execution.

For fare, the pipeline calculates:

* mean
* median
* mode

The skewness interpretation compares their ordering.

A mean substantially greater than the median generally indicates a right-skewed distribution, while the reverse ordering indicates left skew.

---

# Bivariate Analysis

The following survival rates are calculated:

1. Survival by sex
2. Survival by passenger class
3. Survival by sex and passenger class

Boolean masking is also demonstrated using expressions such as:

```python
df.loc[
    (df["sex"] == "female")
    &
    (df["pclass"] == 1)
]
```

The correlation matrix contains exactly:

```text
survived
pclass
age
sibsp
parch
fare
```

The derived boolean columns `adult_male` and `alone` are intentionally excluded.

The two strongest correlations are determined by ranking all off-diagonal correlation coefficients according to their absolute values.

---

# Multivariate Data Story

At least four charts are generated:

### 1. Survival by sex

This demonstrates the strong difference in survival outcomes between male and female passengers.

### 2. Survival by passenger class

This demonstrates the relationship between passenger class and survival.

### 3. Survival by sex and passenger class

Combining the two dimensions reveals that sex and passenger class together provide a stronger explanation of survival patterns than either variable alone.

### 4. Age, fare and survival

The scatter plot examines the interaction between age and fare while using survival as an additional dimension.

Together, these charts support the conclusion that survival was strongly associated with passenger sex and socioeconomic position represented by passenger class and fare.

---

# Standardization Sanity Check

Age and fare are standardized using:

```text
z = (x - mean) / standard deviation
```

The pipeline prints the means and standard deviations before and after transformation.

The transformed variables should have approximately:

```text
mean = 0
standard deviation = 1
```

This transformation is only an exploratory sanity check.

It is not reused by the modeling pipeline.

---

# Classification

The target variable is:

```text
survived
```

The following features are used:

```text
pclass
sex
age
sibsp
parch
fare
embarked
```

The data is split into training and testing sets using a stratified split.

Stratification is important because it preserves approximately the same survived/not-survived class proportions in both datasets.

---

# Modeling Preprocessing

All preprocessing is performed inside a scikit-learn pipeline.

Numeric features use:

```text
Median Imputation
       ↓
StandardScaler
```

Categorical features use:

```text
Most-Frequent Imputation
       ↓
OneHotEncoder
```

The preprocessing pipeline is fitted only on the training data.

The test data is passed through the already-fitted preprocessing pipeline without fitting any transformation on the test data.

This prevents test-set information from leaking into model training.

---

# Classification Models

Three models are trained using the same train/test split:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The Decision Tree is visualized using `plot_tree`, including feature names and class names.

---

# Classification Metrics

Every classifier is evaluated using:

* Confusion matrix
* Accuracy
* Precision
* Recall
* F1 score
* ROC curve
* AUC

The results are saved to:

```text
outputs/classification_model_comparison.csv
```

---

# Imbalance Handling

Three approaches are compared:

1. Baseline Logistic Regression
2. Logistic Regression with `class_weight="balanced"`
3. Logistic Regression with SMOTE

SMOTE is applied only to the training portion through an imbalanced-learn pipeline.

The test set remains untouched so that evaluation represents performance on the original class distribution.

The comparison is saved to:

```text
outputs/imbalance_comparison.csv
```

F1 score is used as the primary comparison criterion because it balances precision and recall.

---

# Random Forest Hyperparameter Tuning

`GridSearchCV` is used to tune:

```text
n_estimators
max_depth
max_features
```

The Random Forest is constructed with:

```python
RandomForestClassifier(
    oob_score=True,
    bootstrap=True
)
```

This allows the out-of-bag score to be reported after fitting.

The best parameter combination, cross-validation score and OOB score are printed.

---

# Regression Side Task

A multivariate linear regression model predicts:

```text
fare
```

from:

```text
survived
pclass
age
sibsp
parch
```

The following metrics are reported:

* MAE
* RMSE
* R²
* Adjusted R²

A residual plot is generated at:

```text
outputs/regression_residuals.png
```

The residual plot is inspected for systematic changes in residual spread across predicted values.

---

# Model Comparison

Classification and regression metrics are kept as separate metric groups because they represent different modeling tasks and are not directly comparable.

Classification metrics:

```text
Accuracy
Precision
Recall
F1
AUC
```

Regression metrics:

```text
MAE
RMSE
R²
Adjusted R²
```

The recommended classifier is selected using the classification results, with particular attention to F1 and AUC rather than accuracy alone.

The exact recommendation and metric values are printed after execution because they depend on the resulting train/test split and model performance.

---

# Model Persistence

The complete fitted classification pipeline is saved using:

```python
joblib.dump(
    full_pipeline,
    "artifacts/best_pipeline.joblib"
)
```

The saved object contains both:

```text
Preprocessing
    +
Final Estimator
```

Therefore the artifact can accept raw input data without manually performing imputation, encoding or scaling.

The pipeline is reloaded using:

```python
joblib.load(...)
```

and tested on raw passenger input.

---

# Reproducibility

The pipeline uses:

```text
random_state = 42
```

for stochastic model and data-splitting operations where applicable.

The generated CSV, charts and model artifact are reproducible by running:

```bash
python analytics_pipeline.py
```

---

# Output Files

```text
outputs/
├── age_histogram.png
├── age_boxplot.png
├── fare_histogram.png
├── fare_boxplot.png
├── correlation_heatmap.png
├── survival_by_sex.png
├── survival_by_class.png
├── survival_sex_class.png
├── age_fare_survival.png
├── age_standardization.png
├── fare_standardization.png
├── decision_tree.png
├── confusion_matrices.png
├── roc_curves.png
├── regression_residuals.png
├── classification_model_comparison.csv
├── imbalance_comparison.csv
└── regression_model_metrics.csv

artifacts/
└── best_pipeline.joblib
```
