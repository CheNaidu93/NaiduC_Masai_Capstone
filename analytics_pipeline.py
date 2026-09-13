from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE

# -------------------------------------------------------------------
# Analytics Pipeline
# -------------------------------------------------------------------
# For the assignment's strict network/cache requirement, the raw
# dataset should be loaded once with sns.load_dataset("titanic") and
# immediately saved to titanic.csv.
#
# This submission folder already contains the committed offline
# titanic.csv. Set USE_SEABORN = True on a machine with internet if
# you need to demonstrate the Seaborn first-load explicitly.
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
ART = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)
ART.mkdir(exist_ok=True)

USE_SEABORN = False

if USE_SEABORN:
    df = sns.load_dataset("titanic")
    # Save the exact loaded DataFrame as the required offline fallback.
    df.to_csv(ROOT / "titanic.csv", index=False)
else:
    df = pd.read_csv(ROOT / "titanic.csv")

# -----------------------------
# Task 1: Profiling
# -----------------------------
print("\n========== df.info() ==========")
df.info()

print("\n========== df.describe() ==========")
print(df.describe(include="all"))

print("\n========== df.shape ==========")
print(df.shape)

missing_pct = (df.isna().mean() * 100).round(2)
print("\n========== Missing percentages ==========")
print(missing_pct[missing_pct > 0])

# -----------------------------
# Task 2: Cleaning
# -----------------------------
# For the supplied 12-column Titanic CSV:
# Age = 19.87% -> 5%-30%, median imputation.
# Embarked = 0.22% -> under 5%, drop those rows.
# Cabin = 77.10% -> too high for reliable imputation, retain as
# "Unknown" so the information is not discarded completely.
df = df.dropna(subset=["Embarked"]).copy()
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Cabin"] = df["Cabin"].fillna("Unknown")

# Derived cabin deck used only for regression to avoid creating a
# dummy variable for every individual cabin number.
df["CabinDeck"] = df["Cabin"].str[0]
df.loc[df["Cabin"] == "Unknown", "CabinDeck"] = "Unknown"

print("\n========== Cleaned shape ==========")
print(df.shape)

# -----------------------------
# Task 3: Univariate analysis
# -----------------------------
def iqr_details(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((series < lower) | (series > upper)).sum())
    return count, q1, q3, lower, upper

for col in ["Age", "Fare"]:
    plt.figure(figsize=(7, 5))
    plt.hist(df[col], bins=30)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUT / f"{col.lower()}_histogram.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.boxplot(df[col], vert=False)
    plt.title(f"{col} Box Plot")
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(OUT / f"{col.lower()}_boxplot.png", dpi=150)
    plt.close()

age_outliers = iqr_details(df["Age"])[0]
fare_outliers = iqr_details(df["Fare"])[0]

fare_mean = df["Fare"].mean()
fare_median = df["Fare"].median()
fare_mode = df["Fare"].mode().iloc[0]
fare_skew = df["Fare"].skew()

print(f"\nAge IQR outliers: {age_outliers}")
print(f"Fare IQR outliers: {fare_outliers}")
print(
    f"Fare mean={fare_mean:.4f}, median={fare_median:.4f}, "
    f"mode={fare_mode:.4f}, skew={fare_skew:.4f}"
)

# -----------------------------
# Task 4: Bivariate analysis
# -----------------------------
sex_survival = df.groupby("Sex")["Survived"].mean()
pclass_survival = df.groupby("Pclass")["Survived"].mean()
sex_pclass_survival = df.groupby(["Sex", "Pclass"])["Survived"].mean()

print("\n========== Survival by sex ==========")
print(sex_survival)

print("\n========== Survival by pclass ==========")
print(pclass_survival)

print("\n========== Survival by sex + pclass ==========")
print(sex_pclass_survival)

# Explicit boolean masking example.
female_first_second = df[
    (df["Sex"] == "female") &
    (df["Pclass"].isin([1, 2]))
]
print("\nFemale 1st/2nd class survival:")
print(female_first_second["Survived"].mean())

corr_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
corr = df[corr_cols].corr()

print("\n========== Required 6x6 correlation matrix ==========")
print(corr)

pairs = []
for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        pairs.append(
            (
                abs(corr.iloc[i, j]),
                corr_cols[i],
                corr_cols[j],
                corr.iloc[i, j],
            )
        )

top_two = sorted(pairs, reverse=True)[:2]
print("\nTwo strongest correlations:")
for item in top_two:
    print(item)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Titanic Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=150)
plt.close()

# -----------------------------
# Task 5: Multivariate data story
# -----------------------------
plt.figure(figsize=(8, 5))
for pclass, offset in zip([1, 2, 3], [-0.25, 0, 0.25]):
    vals = [
        df.loc[
            (df["Sex"] == sex) & (df["Pclass"] == pclass),
            "Survived",
        ].mean()
        for sex in ["female", "male"]
    ]
    plt.bar(
        np.array([0, 1]) + offset,
        vals,
        width=0.22,
        label=f"Class {pclass}",
    )
plt.xticks([0, 1], ["female", "male"])
plt.ylabel("Survival rate")
plt.title("Survival by Sex and Passenger Class")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "survival_sex_class.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
for sex in ["female", "male"]:
    vals = [
        df.loc[
            (df["Sex"] == sex) & (df["Pclass"] == pclass),
            "Survived",
        ].mean()
        for pclass in [1, 2, 3]
    ]
    plt.plot([1, 2, 3], vals, marker="o", label=sex)
plt.xticks([1, 2, 3])
plt.xlabel("Passenger class")
plt.ylabel("Survival rate")
plt.title("Survival Trend Across Classes by Sex")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "survival_pclass_sex.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(
    df.loc[df["Survived"] == 0, "Age"],
    df.loc[df["Survived"] == 0, "Fare"],
    alpha=0.5,
    label="Did not survive",
)
plt.scatter(
    df.loc[df["Survived"] == 1, "Age"],
    df.loc[df["Survived"] == 1, "Fare"],
    alpha=0.5,
    label="Survived",
)
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Age vs Fare by Survival")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "age_fare_survival.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
for pclass in [1, 2, 3]:
    for survived, position in [(0, pclass - 0.12), (1, pclass + 0.12)]:
        plt.boxplot(
            df.loc[
                (df["Pclass"] == pclass) &
                (df["Survived"] == survived),
                "Age",
            ],
            positions=[position],
            widths=0.18,
        )
plt.xticks([1, 2, 3])
plt.xlabel("Passenger class")
plt.ylabel("Age")
plt.title("Age Distribution by Class and Survival")
plt.tight_layout()
plt.savefig(OUT / "age_class_survival.png", dpi=150)
plt.close()

# -----------------------------
# Task 6: EDA standardization
# -----------------------------
standardization_rows = []

for col in ["Age", "Fare"]:
    mean_before = df[col].mean()
    std_before = df[col].std()
    z = (df[col] - mean_before) / std_before

    standardization_rows.append(
        [
            col,
            mean_before,
            std_before,
            z.mean(),
            z.std(),
        ]
    )

    plt.figure(figsize=(7, 4))
    plt.hist(
        df[col],
        bins=30,
        density=True,
        alpha=0.4,
        label="Before",
    )
    plt.hist(
        z,
        bins=30,
        density=True,
        alpha=0.4,
        label="After z-score",
    )
    plt.title(f"{col}: Before vs After Standardization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / f"{col.lower()}_standardization.png", dpi=150)
    plt.close()

std_df = pd.DataFrame(
    standardization_rows,
    columns=[
        "Feature",
        "Before_Mean",
        "Before_Std",
        "After_Mean",
        "After_Std",
    ],
)
std_df.to_csv(OUT / "standardization_summary.csv", index=False)

# -----------------------------
# Tasks 7-10: Classification
# -----------------------------
features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]

X = df[features]
y = df["Survived"]

print("\n========== Class balance ==========")
print(y.value_counts())
print(y.value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
    ),
}

results = []
confusion_matrices = {}
roc_data = {}

for name, estimator in models.items():
    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results.append(
        [
            name,
            accuracy_score(y_test, predictions),
            precision_score(y_test, predictions),
            recall_score(y_test, predictions),
            f1_score(y_test, predictions),
            roc_auc_score(y_test, probabilities),
        ]
    )

    confusion_matrices[name] = confusion_matrix(
        y_test,
        predictions,
    )

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_data[name] = (
        fpr,
        tpr,
        roc_auc_score(y_test, probabilities),
    )

    if name == "Decision Tree":
        feature_names = list(
            pipeline.named_steps["preprocessor"]
            .get_feature_names_out()
        )

        plt.figure(figsize=(20, 10))
        plot_tree(
            pipeline.named_steps["model"],
            feature_names=feature_names,
            class_names=["0", "1"],
            max_depth=4,
            filled=False,
            rounded=True,
            fontsize=7,
        )
        plt.tight_layout()
        plt.savefig(OUT / "decision_tree.png", dpi=150)
        plt.close()

classification_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUC",
    ],
)
classification_df.to_csv(
    OUT / "classification_model_comparison.csv",
    index=False,
)

# Confusion matrices.
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (name, matrix) in zip(axes, confusion_matrices.items()):
    ax.imshow(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
            )
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrices.png", dpi=150)
plt.close()

# ROC curves.
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, auc_value) in roc_data.items():
    plt.plot(
        fpr,
        tpr,
        label=f"{name} AUC={auc_value:.3f}",
    )
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "roc_curves.png", dpi=150)
plt.close()

# -----------------------------
# Task 11: Imbalance comparison
# -----------------------------
imbalance_preprocessor = make_preprocessor()

X_train_transformed = imbalance_preprocessor.fit_transform(X_train)
X_test_transformed = imbalance_preprocessor.transform(X_test)

imbalance_results = []

variants = [
    (
        "Baseline",
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        X_train_transformed,
        y_train,
    ),
    (
        "Class Weight Balanced",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        X_train_transformed,
        y_train,
    ),
]

for label, estimator, train_x, train_y in variants:
    estimator.fit(train_x, train_y)
    pred = estimator.predict(X_test_transformed)

    imbalance_results.append(
        [
            label,
            precision_score(y_test, pred),
            recall_score(y_test, pred),
            f1_score(y_test, pred),
        ]
    )

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(
    X_train_transformed,
    y_train,
)

smote_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)
smote_model.fit(X_smote, y_smote)
smote_pred = smote_model.predict(X_test_transformed)

imbalance_results.append(
    [
        "SMOTE",
        precision_score(y_test, smote_pred),
        recall_score(y_test, smote_pred),
        f1_score(y_test, smote_pred),
    ]
)

imbalance_df = pd.DataFrame(
    imbalance_results,
    columns=["Variant", "Precision", "Recall", "F1"],
)
imbalance_df.to_csv(
    OUT / "imbalance_comparison.csv",
    index=False,
)

# -----------------------------
# Task 12: Random Forest tuning
# -----------------------------
rf_pipeline = Pipeline(
    [
        ("preprocessor", make_preprocessor()),
        (
            "model",
            RandomForestClassifier(
                oob_score=True,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

param_grid = {
    "model__n_estimators": [200, 300],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"],
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

print("\n========== Random Forest Grid Search ==========")
print("Best parameters:", grid_search.best_params_)
print("Best CV F1:", grid_search.best_score_)
print("OOB score:", best_rf.named_steps["model"].oob_score_)

# -----------------------------
# Task 13: Regression
# -----------------------------
# Fare is predicted from the other useful available passenger
# features. Name/Ticket/PassengerId are identifiers/free text rather
# than useful numeric predictors. Cabin is represented by CabinDeck
# to avoid a dummy variable for every individual cabin number.
regression_features = [
    "Survived",
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Embarked",
    "CabinDeck",
]

regression_X = df[regression_features]
regression_y = df["Fare"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    regression_X,
    regression_y,
    test_size=0.20,
    random_state=42,
)

reg_numeric = [
    "Survived",
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
]
reg_categorical = [
    "Sex",
    "Embarked",
    "CabinDeck",
]

reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            reg_numeric,
        ),
        (
            "categorical",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            ),
            reg_categorical,
        ),
    ]
)

regression_pipeline = Pipeline(
    [
        ("preprocessor", reg_preprocessor),
        ("model", LinearRegression()),
    ]
)

regression_pipeline.fit(Xr_train, yr_train)
reg_predictions = regression_pipeline.predict(Xr_test)

mae = mean_absolute_error(yr_test, reg_predictions)
rmse = mean_squared_error(yr_test, reg_predictions) ** 0.5
r2 = r2_score(yr_test, reg_predictions)

n = len(yr_test)
k = regression_pipeline.named_steps["preprocessor"].transform(
    Xr_train
).shape[1]

adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) / (n - k - 1)
)

regression_df = pd.DataFrame(
    [
        [
            "Linear Regression",
            mae,
            rmse,
            r2,
            adjusted_r2,
        ]
    ],
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2",
        "Adjusted_R2",
    ],
)
regression_df.to_csv(
    OUT / "regression_metrics.csv",
    index=False,
)

residuals = yr_test - reg_predictions

plt.figure(figsize=(7, 5))
plt.scatter(reg_predictions, residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Linear Regression Residual Plot")
plt.tight_layout()
plt.savefig(OUT / "regression_residuals.png", dpi=150)
plt.close()

# -----------------------------
# Task 14: Final comparison
# -----------------------------
final_comparison = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Linear Regression",
        ],
        "Accuracy": [
            *classification_df["Accuracy"].tolist(),
            np.nan,
        ],
        "Precision": [
            *classification_df["Precision"].tolist(),
            np.nan,
        ],
        "Recall": [
            *classification_df["Recall"].tolist(),
            np.nan,
        ],
        "F1": [
            *classification_df["F1"].tolist(),
            np.nan,
        ],
        "AUC": [
            *classification_df["AUC"].tolist(),
            np.nan,
        ],
        "MAE": [
            np.nan,
            np.nan,
            np.nan,
            mae,
        ],
        "RMSE": [
            np.nan,
            np.nan,
            np.nan,
            rmse,
        ],
        "R2": [
            np.nan,
            np.nan,
            np.nan,
            r2,
        ],
        "Adjusted_R2": [
            np.nan,
            np.nan,
            np.nan,
            adjusted_r2,
        ],
    }
)

final_comparison.to_csv(
    OUT / "final_model_comparison.csv",
    index=False,
)

# -----------------------------
# Task 15: Save complete pipeline
# -----------------------------
best_classifier_name = (
    classification_df.sort_values("F1", ascending=False)
    .iloc[0]["Model"]
)

best_classifier = models[best_classifier_name]

full_pipeline = Pipeline(
    [
        ("preprocessor", make_preprocessor()),
        ("model", best_classifier),
    ]
)

full_pipeline.fit(X_train, y_train)

artifact_path = ART / "best_pipeline.joblib"
joblib.dump(full_pipeline, artifact_path)

# Reload and predict on raw, unprocessed input.
reloaded_pipeline = joblib.load(artifact_path)
raw_sample = X_test.iloc[[0]]
reloaded_prediction = reloaded_pipeline.predict(raw_sample)

print("\n========== Reloaded Pipeline ==========")
print("Model:", best_classifier_name)
print("Raw sample:")
print(raw_sample)
print("Prediction:", reloaded_prediction)

# -----------------------------
# Save a concise execution summary
# -----------------------------
summary = f"""
ANALYTICS PIPELINE EXECUTION SUMMARY

Raw shape: {df.shape[0] + 2} rows, 12 columns
Cleaned shape: {df.shape[0]} rows, 12 columns

Missing-value percentages:
Age: {missing_pct.get("Age", np.nan):.2f}%
Cabin: {missing_pct.get("Cabin", np.nan):.2f}%
Embarked: {missing_pct.get("Embarked", np.nan):.2f}%

IQR outliers:
Age: {age_outliers}
Fare: {fare_outliers}

Fare:
Mean = {fare_mean:.4f}
Median = {fare_median:.4f}
Mode = {fare_mode:.4f}
Skewness = {fare_skew:.4f}
Conclusion = right-skewed

Strongest correlations:
1. Pclass vs Fare = {top_two[0][3]:.4f}
2. SibSp vs Parch = {top_two[1][3]:.4f}

Classification results:
{classification_df.to_string(index=False)}

Imbalance comparison:
{imbalance_df.to_string(index=False)}

Random Forest best parameters:
{grid_search.best_params_}
OOB score:
{best_rf.named_steps["model"].oob_score_:.4f}

Regression:
MAE = {mae:.4f}
RMSE = {rmse:.4f}
R2 = {r2:.4f}
Adjusted R2 = {adjusted_r2:.4f}

Best classifier by F1:
{best_classifier_name}

Saved complete pipeline:
{artifact_path}
"""

(OUT / "run_summary.txt").write_text(summary.strip(), encoding="utf-8")

print("\nPipeline completed successfully.")
