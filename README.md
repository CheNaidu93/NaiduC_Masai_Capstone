# Module 2 — Analytics Pipeline

This module implements the Titanic profiling, cleaning, EDA,
classification, imbalance comparison, Random Forest tuning,
regression, model comparison and model persistence requirements.

The assignment requires the dataset to be loaded once through
`seaborn.load_dataset("titanic")`, immediately saved as
`titanic.csv`, and then reused throughout the module. fileciteturn4file0L13-L17

## Important execution note

The supplied `titanic.csv` in this submission is the classic
891-row Titanic CSV with 12 columns. The execution results below
were produced from that supplied CSV because the execution
environment used for this preparation could not reach Seaborn's
online repository.

For strict local demonstration of the assignment's first-load
requirement, set `USE_SEABORN = True` in `analytics_pipeline.py`
and run it once on a machine with internet access. The script
contains only one `sns.load_dataset("titanic")` call and saves
the resulting DataFrame immediately to `titanic.csv`.

## Run

```bash
cd analytics
pip install -r requirements.txt
python analytics_pipeline.py
python predict.py
```

## Cleaning decisions

The supplied raw dataset has:

- Age: 19.87% missing → 5%–30% threshold → median imputation.
- Embarked: 0.22% missing → under 5% threshold → drop those rows.
- Cabin: 77.10% missing → too high for reliable imputation → retain
  the information by encoding missing values as `Unknown`.

The cleaned dataset therefore contains 889 rows.

The assignment specifies these percentage-based missing-value
decisions and requires the measured percentage to be stated before
the strategy is chosen. fileciteturn4file0L13-L15

## Task 3 — Univariate results

IQR outliers:

- Age: 65
- Fare: 114

Fare statistics:

- Mean: 32.0967
- Median: 14.4542
- Mode: 8.0500
- Skewness: 4.8014

Because mean > median > mode and the skewness is strongly positive,
Fare is right-skewed.

## Task 4 — Bivariate results

### Survival by sex

- Female: 74.04%
- Male: 18.89%

### Survival by passenger class

- Class 1: 62.62%
- Class 2: 47.28%
- Class 3: 24.24%

### Survival by sex and class

| Sex | Pclass | Survival |
|---|---:|---:|
| Female | 1 | 96.74% |
| Female | 2 | 92.11% |
| Female | 3 | 50.00% |
| Male | 1 | 36.89% |
| Male | 2 | 15.74% |
| Male | 3 | 13.54% |

The required correlation matrix uses exactly:
`Survived`, `Pclass`, `Age`, `SibSp`, `Parch`, and `Fare`.
The boolean derived fields `adult_male` and `alone` are excluded as
required. fileciteturn4file0L16-L18

### Two strongest correlations

1. `Pclass` vs `Fare`: -0.5482
2. `SibSp` vs `Parch`: +0.4145

The negative Pclass/Fare correlation means lower numerical class
values (higher passenger class) are associated with higher fares.
The positive SibSp/Parch relationship indicates passengers traveling
with siblings/spouses also tended to travel with parents/children,
reflecting family-group structure.

## Task 5 — Multivariate data story

### Chart 1 — Survival by Sex and Passenger Class

Female passengers have substantially higher survival rates than
male passengers across every passenger class. The effect is strongest
in first and second class, while third-class female survival is still
much higher than male survival.

### Chart 2 — Survival Trend Across Classes by Sex

Survival decreases as passenger class moves from first to third class
for both sexes. The decline is particularly pronounced among male
passengers, showing that sex and class together provide a stronger
survival story than either feature alone.

### Chart 3 — Age vs Fare by Survival

Survivors are more concentrated among higher-fare passengers, while
many non-survivors appear in the lower-fare region. The plot also
shows that survival is not explained by age alone and interacts with
other passenger characteristics.

### Chart 4 — Age Distribution by Class and Survival

Age distributions differ across passenger classes and survival
outcomes. Younger passengers appear in all classes, but class and
survival together reveal different age patterns rather than a single
age threshold separating survivors from non-survivors.

The assignment requires at least four distinct multivariate charts,
each with its own written 2–4 sentence interpretation. fileciteturn4file0L17-L18

## Task 6 — Standardization

The full cleaned DataFrame was used only for the EDA-stage
standardization check.

For both Age and Fare, the z-score transformation produced
approximately mean 0 and standard deviation 1. This transformation
is not reused in modeling; the modeling pipeline fits its own
StandardScaler only on the training split, as required. fileciteturn4file0L20-L23

## Task 7–10 — Classification

The cleaned data has:

- Not survived: 549 (61.75%)
- Survived: 340 (38.25%)

A stratified 80/20 train/test split with `random_state=42` was used.
Stratification preserves approximately the same survived/not-survived
class proportions in train and test.

### Classification comparison

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree | 0.7640 | 0.7600 | 0.5588 | 0.6441 | 0.8374 |
| Random Forest | 0.8034 | 0.7619 | 0.7059 | 0.7328 | 0.8237 |

All preprocessing is implemented through a
`ColumnTransformer`/`Pipeline`, so imputers, encoders and scalers are
fit only during training and then applied to the test data in
transform-only mode. The assignment explicitly requires this
leakage-safe structure. fileciteturn4file0L22-L25

## Task 11 — Imbalance comparison

| Variant | Precision | Recall | F1 |
|---|---:|---:|---:|
| Baseline | 0.7833 | 0.6912 | 0.7344 |
| Class Weight Balanced | 0.7183 | 0.7500 | 0.7338 |
| SMOTE | 0.7353 | 0.7353 | 0.7353 |

SMOTE produced the highest F1 in this comparison, although the
difference from baseline is very small. Class weighting increased
recall but reduced precision. SMOTE was applied only after the
training transformation and only to the training fold, avoiding test
data leakage. fileciteturn4file0L26-L27

## Task 12 — Random Forest tuning

Best GridSearchCV parameters:

```text
max_depth = None
max_features = sqrt
n_estimators = 300
```

Best cross-validation F1: approximately 0.7449.

OOB score: approximately 0.8073.

The Random Forest estimator was created with `oob_score=True`, which
is required for `oob_score_` to be available. fileciteturn4file0L27-L27

## Task 13 — Regression

Fare was predicted using the other useful passenger features:
Survived, Pclass, Sex, Age, SibSp, Parch, Embarked and CabinDeck.
PassengerId, Name and Ticket were treated as identifiers/free-text
rather than meaningful predictive variables.

Regression results:

- MAE: 16.9074
- RMSE: 37.8023
- R²: 0.4644
- Adjusted R²: 0.4000

The residual plot shows a wider residual spread at higher predicted
fares, indicating likely heteroscedasticity rather than a constant
residual variance.

The assignment requires MAE, RMSE, R², Adjusted R² and an explicit
heteroscedasticity conclusion. fileciteturn4file0L28-L29

## Task 14 — Final recommendation

I would deploy **Logistic Regression** for this dataset. It has the
highest accuracy (0.8090) and highest AUC (0.8610), while its F1 score
(0.7344) is almost identical to Random Forest (0.7328). Random Forest
has slightly higher recall (0.7059 vs 0.6912), but Logistic Regression
provides the strongest overall combination of accuracy, precision,
F1 and especially ranking performance measured by AUC. The Decision
Tree is weaker across all major classification metrics, so it would
not be my first deployment choice.

Classification and regression metrics are intentionally presented
as separate metric groups because they are not directly comparable
numeric scales. fileciteturn4file0L29-L30

## Task 15 — Saved pipeline

The complete fitted classification pipeline is saved as:

```text
artifacts/best_pipeline.joblib
```

The saved object contains both preprocessing and the final estimator,
not just the bare model. `predict.py` reloads the object with
`joblib.load()` and sends raw passenger fields directly to it.

This satisfies the requirement that the persisted artifact be usable
end-to-end on raw, unpreprocessed input. fileciteturn4file0L30-L30

## Generated artifacts

The `outputs/` directory contains the required charts and metric
tables, including:

- age/fare histograms and box plots
- correlation heatmap
- four multivariate charts
- standardization checks
- decision tree visualization
- confusion matrices
- ROC curves
- imbalance comparison
- Random Forest tuning summary
- regression residual plot
- classification comparison
- regression metrics
- final model comparison

The saved CSV and artifacts are included so the repository contains
the evidence needed for grading.
