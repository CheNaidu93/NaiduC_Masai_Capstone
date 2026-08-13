"""
Zepto Capstone - Module 2
Analytics Pipeline

This script performs:

1. Titanic dataset loading
2. Offline CSV creation
3. Data profiling
4. Missing-value analysis
5. Data cleaning
6. Univariate analysis
7. Bivariate analysis
8. Correlation analysis
9. Multivariate visualization
10. Standardization sanity check
11. Stratified train/test split
12. ML preprocessing pipeline
13. Logistic Regression
14. Decision Tree
15. Random Forest
16. Classification evaluation
17. Imbalance comparison
18. Random Forest GridSearchCV
19. OOB evaluation
20. Linear regression
21. Model comparison
22. Complete pipeline persistence
23. Pipeline reload validation
"""

import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "titanic.csv"
)

PIPELINE_PATH = os.path.join(
    ARTIFACT_DIR,
    "best_pipeline.joblib"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    ARTIFACT_DIR,
    exist_ok=True
)


RANDOM_STATE = 42


# ============================================================
# HELPER
# ============================================================

def save_plot(filename):
    """
    Save current matplotlib figure.
    """

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PART A
# TASK 1
# LOAD DATASET ONCE
# ============================================================

def load_dataset_once():

    print("\n" + "=" * 80)
    print("TASK 1 - LOADING TITANIC DATASET")
    print("=" * 80)

    if os.path.exists(CSV_PATH):

        print(
            "titanic.csv already exists."
        )

        print(
            "Loading offline fallback."
        )

        df = pd.read_csv(
            CSV_PATH
        )

    else:

        print(
            "Loading Titanic dataset from "
            "Seaborn."
        )

        df = sns.load_dataset(
            "titanic"
        )

        print(
            "Saving offline fallback:"
        )

        df.to_csv(
            CSV_PATH,
            index=False
        )

    return df


# ============================================================
# TASK 1 - PROFILING
# ============================================================

def profile_dataset(df):

    print("\n" + "=" * 80)
    print("DATASET PROFILE")
    print("=" * 80)

    print("\nShape:")
    print(df.shape)

    print("\nINFO:")
    df.info()

    print("\nDESCRIBE:")
    print(df.describe(include="all"))

    print(
        "\nMissing-value percentages:"
    )

    missing = (
        df.isnull()
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    for column, percentage in missing.items():

        print(
            f"{column}: "
            f"{percentage:.2f}%"
        )

    return missing


# ============================================================
# TASK 2
# MISSING VALUE HANDLING
# ============================================================

def clean_dataset(df, missing_percentages):

    print("\n" + "=" * 80)
    print("TASK 2 - CLEANING")
    print("=" * 80)

    df = df.copy()

    # --------------------------------------------------------
    # DISPLAY STRATEGY
    # --------------------------------------------------------

    print("\nCleaning decisions:")

    for column, percentage in missing_percentages.items():

        if percentage < 5:

            strategy = (
                "Drop rows because missingness "
                "is below 5%."
            )

        elif percentage <= 30:

            strategy = (
                "Impute because missingness "
                "is between 5% and 30%."
            )

        else:

            strategy = (
                "Use a missing category or "
                "drop the column because "
                "missingness is above 30%."
            )

        print(
            f"{column}: "
            f"{percentage:.2f}% -> "
            f"{strategy}"
        )

    # --------------------------------------------------------
    # TITANIC-SPECIFIC CLEANING
    # --------------------------------------------------------

    # embarked has very small missing percentage.
    if "embarked" in df.columns:

        embarked_missing = (
            df["embarked"].isna().mean()
            * 100
        )

        if embarked_missing < 5:

            df = df.dropna(
                subset=["embarked"]
            )

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    if "age" in df.columns:

        age_missing = (
            df["age"].isna().mean()
            * 100
        )

        if 5 <= age_missing <= 30:

            df["age"] = (
                df["age"]
                .fillna(
                    df["age"].median()
                )
            )

    # --------------------------------------------------------
    # DECK
    # --------------------------------------------------------

    if "deck" in df.columns:

        deck_missing = (
            df["deck"].isna().mean()
            * 100
        )

        if deck_missing > 30:

            # Deck has very high missingness.
            # We preserve information by treating
            # missingness as its own category.

            df["deck"] = (
                df["deck"]
                .astype("object")
                .fillna("Unknown")
            )

    # --------------------------------------------------------
    # EMBARK_TOWN
    # --------------------------------------------------------

    if "embark_town" in df.columns:

        df["embark_town"] = (
            df["embark_town"]
            .fillna("Unknown")
        )

    # --------------------------------------------------------
    # FARE
    # --------------------------------------------------------

    if "fare" in df.columns:

        fare_missing = (
            df["fare"].isna().mean()
            * 100
        )

        if fare_missing > 0:

            df["fare"] = (
                df["fare"]
                .fillna(
                    df["fare"].median()
                )
            )

    print(
        "\nRemaining missing values:"
    )

    print(
        df.isnull().sum()
    )

    return df


# ============================================================
# TASK 3
# UNIVARIATE ANALYSIS
# ============================================================

def iqr_outliers(series):

    q1 = series.quantile(
        0.25
    )

    q3 = series.quantile(
        0.75
    )

    iqr = q3 - q1

    lower = (
        q1 - 1.5 * iqr
    )

    upper = (
        q3 + 1.5 * iqr
    )

    mask = (
        (series < lower)
        |
        (series > upper)
    )

    return (
        mask.sum(),
        lower,
        upper
    )


def univariate_analysis(df):

    print("\n" + "=" * 80)
    print("TASK 3 - UNIVARIATE ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # AGE HISTOGRAM
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        df["age"],
        kde=True
    )

    plt.title(
        "Age Distribution"
    )

    plt.xlabel(
        "Age"
    )

    plt.ylabel(
        "Count"
    )

    save_plot(
        "age_histogram.png"
    )

    # --------------------------------------------------------
    # AGE BOXPLOT
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 4)
    )

    sns.boxplot(
        x=df["age"]
    )

    plt.title(
        "Age Box Plot"
    )

    save_plot(
        "age_boxplot.png"
    )

    # --------------------------------------------------------
    # FARE HISTOGRAM
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        df["fare"],
        kde=True
    )

    plt.title(
        "Fare Distribution"
    )

    plt.xlabel(
        "Fare"
    )

    plt.ylabel(
        "Count"
    )

    save_plot(
        "fare_histogram.png"
    )

    # --------------------------------------------------------
    # FARE BOXPLOT
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 4)
    )

    sns.boxplot(
        x=df["fare"]
    )

    plt.title(
        "Fare Box Plot"
    )

    save_plot(
        "fare_boxplot.png"
    )

    # --------------------------------------------------------
    # IQR
    # --------------------------------------------------------

    age_outliers, age_lower, age_upper = (
        iqr_outliers(df["age"])
    )

    fare_outliers, fare_lower, fare_upper = (
        iqr_outliers(df["fare"])
    )

    print(
        f"\nAge outliers: {age_outliers}"
    )

    print(
        f"Age bounds: "
        f"{age_lower:.2f} to "
        f"{age_upper:.2f}"
    )

    print(
        f"Fare outliers: {fare_outliers}"
    )

    print(
        f"Fare bounds: "
        f"{fare_lower:.2f} to "
        f"{fare_upper:.2f}"
    )

    # --------------------------------------------------------
    # FARE STATISTICS
    # --------------------------------------------------------

    fare_mean = df["fare"].mean()

    fare_median = df["fare"].median()

    fare_mode = (
        df["fare"]
        .mode()
        .iloc[0]
    )

    print(
        f"\nFare mean: "
        f"{fare_mean:.4f}"
    )

    print(
        f"Fare median: "
        f"{fare_median:.4f}"
    )

    print(
        f"Fare mode: "
        f"{fare_mode:.4f}"
    )

    if (
        fare_mean
        >
        fare_median
        >
        fare_mode
    ):

        skew_description = (
            "Fare is right-skewed."
        )

    elif (
        fare_mean
        <
        fare_median
        <
        fare_mode
    ):

        skew_description = (
            "Fare is left-skewed."
        )

    else:

        skew_description = (
            "Fare is approximately symmetric "
            "or does not follow a strong "
            "mean-median-mode ordering."
        )

    print(
        "\nDistribution interpretation:"
    )

    print(
        skew_description
    )

    return {
        "age_outliers": age_outliers,
        "fare_outliers": fare_outliers,
        "fare_mean": fare_mean,
        "fare_median": fare_median,
        "fare_mode": fare_mode,
        "fare_skew_description":
            skew_description
    }


# ============================================================
# TASK 4
# BIVARIATE ANALYSIS
# ============================================================

def bivariate_analysis(df):

    print("\n" + "=" * 80)
    print("TASK 4 - BIVARIATE ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # SEX
    # --------------------------------------------------------

    sex_survival = (
        df.groupby("sex")["survived"]
        .mean()
        .mul(100)
    )

    print(
        "\nSurvival rate by sex:"
    )

    print(
        sex_survival
    )

    # Boolean masking explicitly
    female_survival = (
        df.loc[
            df["sex"] == "female",
            "survived"
        ].mean()
        * 100
    )

    male_survival = (
        df.loc[
            df["sex"] == "male",
            "survived"
        ].mean()
        * 100
    )

    print(
        "\nBoolean masking:"
    )

    print(
        f"Female: "
        f"{female_survival:.2f}%"
    )

    print(
        f"Male: "
        f"{male_survival:.2f}%"
    )

    # --------------------------------------------------------
    # PCLASS
    # --------------------------------------------------------

    class_survival = (
        df.groupby("pclass")["survived"]
        .mean()
        .mul(100)
    )

    print(
        "\nSurvival rate by pclass:"
    )

    print(
        class_survival
    )

    # --------------------------------------------------------
    # SEX + PCLASS
    # --------------------------------------------------------

    sex_class_survival = (
        df.groupby(
            ["sex", "pclass"]
        )["survived"]
        .mean()
        .mul(100)
        .reset_index()
    )

    print(
        "\nSurvival by sex and pclass:"
    )

    print(
        sex_class_survival
    )

    # --------------------------------------------------------
    # BOOLEAN COMBINATION
    # --------------------------------------------------------

    female_first_class = df.loc[
        (df["sex"] == "female")
        &
        (df["pclass"] == 1)
    ]

    male_third_class = df.loc[
        (df["sex"] == "male")
        &
        (df["pclass"] == 3)
    ]

    print(
        "\nBoolean combination examples:"
    )

    print(
        "Female + first class survival: "
        f"{female_first_class['survived'].mean() * 100:.2f}%"
    )

    print(
        "Male + third class survival: "
        f"{male_third_class['survived'].mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    correlation_columns = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare"
    ]

    correlation_matrix = (
        df[
            correlation_columns
        ]
        .corr()
    )

    print(
        "\nCorrelation matrix:"
    )

    print(
        correlation_matrix
    )

    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 7)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title(
        "Titanic Numeric Feature Correlation"
    )

    save_plot(
        "correlation_heatmap.png"
    )

    # --------------------------------------------------------
    # STRONGEST TWO CORRELATIONS
    # --------------------------------------------------------

    pairs = []

    columns = correlation_matrix.columns

    for i in range(
        len(columns)
    ):

        for j in range(
            i + 1,
            len(columns)
        ):

            feature_a = columns[i]

            feature_b = columns[j]

            correlation = (
                correlation_matrix
                .loc[
                    feature_a,
                    feature_b
                ]
            )

            pairs.append(
                (
                    feature_a,
                    feature_b,
                    correlation,
                    abs(correlation)
                )
            )

    pairs = sorted(
        pairs,
        key=lambda x: x[3],
        reverse=True
    )

    print(
        "\nTwo strongest correlations:"
    )

    for pair in pairs[:2]:

        print(
            f"{pair[0]} vs {pair[1]}: "
            f"{pair[2]:.4f}"
        )

    return {
        "sex_survival":
            sex_survival,

        "class_survival":
            class_survival,

        "sex_class_survival":
            sex_class_survival,

        "correlation":
            correlation_matrix,

        "strongest_correlations":
            pairs[:2]
    }


# ============================================================
# TASK 5
# MULTIVARIATE DATA STORY
# ============================================================

def multivariate_analysis(df):

    print("\n" + "=" * 80)
    print("TASK 5 - MULTIVARIATE DATA STORY")
    print("=" * 80)

    # --------------------------------------------------------
    # CHART 1
    # SEX + SURVIVAL
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    sns.barplot(
        data=df,
        x="sex",
        y="survived"
    )

    plt.title(
        "Survival Rate by Sex"
    )

    plt.ylabel(
        "Survival Rate"
    )

    save_plot(
        "survival_by_sex.png"
    )

    # --------------------------------------------------------
    # CHART 2
    # CLASS + SURVIVAL
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    sns.barplot(
        data=df,
        x="pclass",
        y="survived"
    )

    plt.title(
        "Survival Rate by Passenger Class"
    )

    plt.ylabel(
        "Survival Rate"
    )

    save_plot(
        "survival_by_class.png"
    )

    # --------------------------------------------------------
    # CHART 3
    # SEX + CLASS
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
    )

    sns.barplot(
        data=df,
        x="pclass",
        y="survived",
        hue="sex"
    )

    plt.title(
        "Survival Rate by Sex and Passenger Class"
    )

    plt.ylabel(
        "Survival Rate"
    )

    save_plot(
        "survival_sex_class.png"
    )

    # --------------------------------------------------------
    # CHART 4
    # AGE + FARE + SURVIVAL
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    sns.scatterplot(
        data=df,
        x="age",
        y="fare",
        hue="survived",
        style="sex",
        alpha=0.7
    )

    plt.title(
        "Age, Fare and Survival"
    )

    save_plot(
        "age_fare_survival.png"
    )

    print(
        """
Data story interpretation:

1. Survival differs substantially by sex, with female passengers
   generally showing higher survival rates than male passengers.

2. Passenger class is also strongly associated with survival.
   Higher-class passengers generally experienced better survival
   outcomes than lower-class passengers.

3. The combination of sex and passenger class reveals an even
   clearer pattern: female passengers in higher classes generally
   had particularly high survival rates, while male passengers
   in lower classes had much lower survival rates.

4. The age/fare scatter plot shows that fare and passenger class
   are related to socioeconomic position, while survival is not
   evenly distributed across the age/fare space.
"""
    )


# ============================================================
# TASK 6
# STANDARDIZATION SANITY CHECK
# ============================================================

def standardization_check(df):

    print("\n" + "=" * 80)
    print("TASK 6 - STANDARDIZATION CHECK")
    print("=" * 80)

    standardized = df[
        ["age", "fare"]
    ].copy()

    before = pd.DataFrame(
        {
            "mean": standardized.mean(),
            "std": standardized.std()
        }
    )

    print(
        "\nBefore standardization:"
    )

    print(
        before
    )

    # pandas std uses ddof=1.
    standardized["age"] = (
        (
            standardized["age"]
            - standardized["age"].mean()
        )
        /
        standardized["age"].std()
    )

    standardized["fare"] = (
        (
            standardized["fare"]
            - standardized["fare"].mean()
        )
        /
        standardized["fare"].std()
    )

    after = pd.DataFrame(
        {
            "mean": standardized.mean(),
            "std": standardized.std()
        }
    )

    print(
        "\nAfter standardization:"
    )

    print(
        after
    )

    # Plot before/after
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    sns.histplot(
        df["age"],
        kde=True,
        ax=axes[0]
    )

    axes[0].set_title(
        "Original Age"
    )

    sns.histplot(
        standardized["age"],
        kde=True,
        ax=axes[1]
    )

    axes[1].set_title(
        "Standardized Age"
    )

    plt.tight_layout()

    save_plot(
        "age_standardization.png"
    )

    # Fare
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    sns.histplot(
        df["fare"],
        kde=True,
        ax=axes[0]
    )

    axes[0].set_title(
        "Original Fare"
    )

    sns.histplot(
        standardized["fare"],
        kde=True,
        ax=axes[1]
    )

    axes[1].set_title(
        "Standardized Fare"
    )

    plt.tight_layout()

    save_plot(
        "fare_standardization.png"
    )

    return standardized


# ============================================================
# PART B
# TASK 7
# STRATIFIED TRAIN TEST SPLIT
# ============================================================

def split_data(df):

    print("\n" + "=" * 80)
    print("TASK 7 - STRATIFIED TRAIN/TEST SPLIT")
    print("=" * 80)

    target = "survived"

    # Features used for classification
    features = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]

    X = df[
        features
    ].copy()

    y = df[
        target
    ].copy()

    print(
        "\nOriginal class balance:"
    )

    print(
        y.value_counts(
            normalize=True
        )
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print(
        "\nTraining class balance:"
    )

    print(
        y_train.value_counts(
            normalize=True
        )
    )

    print(
        "\nTesting class balance:"
    )

    print(
        y_test.value_counts(
            normalize=True
        )
    )

    print(
        """
Stratification is used because survival is a binary target and
the train and test sets should retain approximately the same
survival/non-survival proportions as the original dataset.
Without stratification, random sampling could produce a test
set with a distorted class distribution.
"""
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# TASK 8
# PREPROCESSING
# ============================================================

def create_preprocessor():

    numeric_features = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare"
    ]

    categorical_features = [
        "sex",
        "embarked"
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# TASK 9
# TRAIN THREE CLASSIFIERS
# ============================================================

def train_classifiers(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "=" * 80)
    print("TASK 9/10 - CLASSIFIERS")
    print("=" * 80)

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=5,
                random_state=RANDOM_STATE
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE
            )
    }

    trained_models = {}

    predictions = {}

    probabilities = {}

    metrics = []

    for name, estimator in models.items():

        print(
            f"\nTraining {name}..."
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor()
                ),
                (
                    "model",
                    estimator
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        y_pred = pipeline.predict(
            X_test
        )

        y_probability = pipeline.predict_proba(
            X_test
        )[:, 1]

        trained_models[name] = pipeline

        predictions[name] = y_pred

        probabilities[name] = (
            y_probability
        )

        metrics.append(
            classification_metrics(
                name,
                y_test,
                y_pred,
                y_probability
            )
        )

    metrics_df = pd.DataFrame(
        metrics
    )

    print(
        "\nClassification metrics:"
    )

    print(
        metrics_df
    )

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4)
    )

    for ax, name in zip(
        axes,
        trained_models.keys()
    ):

        cm = confusion_matrix(
            y_test,
            predictions[name]
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )

        ax.set_title(
            name
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )

    plt.tight_layout()

    save_plot(
        "confusion_matrices.png"
    )

    # --------------------------------------------------------
    # ROC CURVES
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    for name in trained_models.keys():

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities[name]
        )

        auc = roc_auc_score(
            y_test,
            probabilities[name]
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} AUC={auc:.3f}"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curves"
    )

    plt.legend()

    save_plot(
        "roc_curves.png"
    )

    return (
        trained_models,
        predictions,
        probabilities,
        metrics_df
    )


def classification_metrics(
    name,
    y_true,
    y_pred,
    y_probability
):

    return {
        "model": name,

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "auc":
            roc_auc_score(
                y_true,
                y_probability
            )
    }


# ============================================================
# DECISION TREE VISUALIZATION
# ============================================================

def visualize_decision_tree(
    trained_models
):

    print(
        "\nRendering decision tree..."
    )

    tree_pipeline = (
        trained_models[
            "Decision Tree"
        ]
    )

    preprocessor = (
        tree_pipeline
        .named_steps[
            "preprocessor"
        ]
    )

    model = (
        tree_pipeline
        .named_steps[
            "model"
        ]
    )

    # Get transformed feature names
    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    plt.figure(
        figsize=(25, 15)
    )

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=[
            "Not Survived",
            "Survived"
        ],
        filled=True,
        rounded=True,
        fontsize=8
    )

    plt.title(
        "Decision Tree Classifier"
    )

    save_plot(
        "decision_tree.png"
    )


# ============================================================
# TASK 11
# IMBALANCE HANDLING
# ============================================================

def imbalance_comparison(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "=" * 80)
    print("TASK 11 - IMBALANCE HANDLING")
    print("=" * 80)

    print(
        "\nClass distribution:"
    )

    print(
        pd.DataFrame(
            {
                "count":
                    y_train.value_counts(),

                "percentage":
                    y_train.value_counts(
                        normalize=True
                    ) * 100
            }
        )
    )

    results = []

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    baseline.fit(
        X_train,
        y_train
    )

    baseline_pred = baseline.predict(
        X_test
    )

    results.append(
        {
            "strategy": "Baseline",
            "precision": precision_score(
                y_test,
                baseline_pred,
                zero_division=0
            ),
            "recall": recall_score(
                y_test,
                baseline_pred,
                zero_division=0
            ),
            "f1": f1_score(
                y_test,
                baseline_pred,
                zero_division=0
            )
        }
    )

    # --------------------------------------------------------
    # CLASS WEIGHT BALANCED
    # --------------------------------------------------------

    balanced = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    balanced.fit(
        X_train,
        y_train
    )

    balanced_pred = balanced.predict(
        X_test
    )

    results.append(
        {
            "strategy":
                "class_weight='balanced'",

            "precision":
                precision_score(
                    y_test,
                    balanced_pred,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y_test,
                    balanced_pred,
                    zero_division=0
                ),

            "f1":
                f1_score(
                    y_test,
                    balanced_pred,
                    zero_division=0
                )
        }
    )

    # --------------------------------------------------------
    # SMOTE
    # --------------------------------------------------------

    smote_pipeline = ImbPipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE
                )
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    smote_pipeline.fit(
        X_train,
        y_train
    )

    smote_pred = smote_pipeline.predict(
        X_test
    )

    results.append(
        {
            "strategy": "SMOTE",

            "precision":
                precision_score(
                    y_test,
                    smote_pred,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y_test,
                    smote_pred,
                    zero_division=0
                ),

            "f1":
                f1_score(
                    y_test,
                    smote_pred,
                    zero_division=0
                )
        }
    )

    results_df = pd.DataFrame(
        results
    )

    print(
        "\nImbalance comparison:"
    )

    print(
        results_df
    )

    best_strategy = (
        results_df
        .sort_values(
            "f1",
            ascending=False
        )
        .iloc[0]
    )

    print(
        "\nBest strategy based on F1:"
    )

    print(
        best_strategy[
            "strategy"
        ]
    )

    return results_df


# ============================================================
# TASK 12
# RANDOM FOREST GRID SEARCH
# ============================================================

def random_forest_tuning(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "=" * 80)
    print("TASK 12 - RANDOM FOREST GRID SEARCH")
    print("=" * 80)

    # Important:
    # OOB score requires oob_score=True.

    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        oob_score=True,
        bootstrap=True
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                rf
            )
        ]
    )

    param_grid = {

        "model__n_estimators": [
            100,
            200
        ],

        "model__max_depth": [
            None,
            5,
            10
        ],

        "model__max_features": [
            "sqrt",
            "log2"
        ]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    print(
        "\nBest parameters:"
    )

    print(
        grid_search.best_params_
    )

    print(
        "\nBest CV score:"
    )

    print(
        grid_search.best_score_
    )

    best_pipeline = (
        grid_search.best_estimator_
    )

    best_rf = (
        best_pipeline
        .named_steps["model"]
    )

    print(
        "\nOOB score:"
    )

    print(
        best_rf.oob_score_
    )

    y_pred = (
        best_pipeline
        .predict(X_test)
    )

    y_probability = (
        best_pipeline
        .predict_proba(X_test)
        [:, 1]
    )

    print(
        "\nTest metrics:"
    )

    print(
        classification_metrics(
            "Tuned Random Forest",
            y_test,
            y_pred,
            y_probability
        )
    )

    return (
        best_pipeline,
        grid_search
    )


# ============================================================
# TASK 13
# REGRESSION
# ============================================================

def regression_task(df):

    print("\n" + "=" * 80)
    print("TASK 13 - LINEAR REGRESSION")
    print("=" * 80)

    regression_features = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch"
    ]

    X = df[
        regression_features
    ].copy()

    y = df[
        "fare"
    ].copy()

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE
        )
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    regression_preprocessor = (
        ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="median"
                                )
                            ),
                            (
                                "scaler",
                                StandardScaler()
                            )
                        ]
                    ),
                    regression_features
                )
            ]
        )
    )

    regression_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                regression_preprocessor
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    )

    regression_pipeline.fit(
        X_train,
        y_train
    )

    predictions = (
        regression_pipeline
        .predict(X_test)
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    n = len(
        y_test
    )

    p = len(
        regression_features
    )

    adjusted_r2 = (
        1
        -
        (
            (1 - r2)
            *
            (n - 1)
            /
            (n - p - 1)
        )
    )

    regression_metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2":
            adjusted_r2
    }

    print(
        "\nRegression metrics:"
    )

    for metric, value in (
        regression_metrics.items()
    ):

        print(
            f"{metric}: "
            f"{value:.4f}"
        )

    # --------------------------------------------------------
    # RESIDUAL PLOT
    # --------------------------------------------------------

    residuals = (
        y_test
        -
        predictions
    )

    plt.figure(
        figsize=(9, 6)
    )

    sns.scatterplot(
        x=predictions,
        y=residuals
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Predicted Fare"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "Linear Regression Residual Plot"
    )

    save_plot(
        "regression_residuals.png"
    )

    print(
        """
Heteroscedasticity assessment:

If the residuals show an approximately constant vertical spread
around zero across predicted values, there is no strong visual
evidence of heteroscedasticity.

If the residual spread increases or decreases systematically as
predicted fare increases, that indicates heteroscedasticity.
"""
    )

    return (
        regression_pipeline,
        regression_metrics
    )


# ============================================================
# TASK 14
# MODEL COMPARISON
# ============================================================

def model_comparison(
    classification_metrics_df,
    regression_metrics
):

    print("\n" + "=" * 80)
    print("TASK 14 - MODEL COMPARISON")
    print("=" * 80)

    print(
        "\nClassification models:"
    )

    print(
        classification_metrics_df
    )

    print(
        "\nRegression model:"
    )

    print(
        pd.DataFrame(
            [regression_metrics]
        )
    )

    classification_metrics_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "classification_model_comparison.csv"
        ),
        index=False
    )

    pd.DataFrame(
        [regression_metrics]
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "regression_model_metrics.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # BEST CLASSIFIER
    # --------------------------------------------------------

    best_classifier = (
        classification_metrics_df
        .sort_values(
            by="f1",
            ascending=False
        )
        .iloc[0]
    )

    print(
        "\nRecommended classifier:"
    )

    print(
        best_classifier["model"]
    )

    print(
        f"""
Recommendation:

The recommended classifier is
{best_classifier['model']} because it achieved an F1 score of
{best_classifier['f1']:.3f}, accuracy of
{best_classifier['accuracy']:.3f}, precision of
{best_classifier['precision']:.3f}, recall of
{best_classifier['recall']:.3f}, and AUC of
{best_classifier['auc']:.3f} on the held-out test set.

The final choice should consider the business cost of false
positives versus false negatives rather than relying only on
accuracy. F1 and AUC provide additional information about the
classifier's overall discriminatory performance.
"""
    )

    return best_classifier


# ============================================================
# TASK 15
# SAVE COMPLETE PIPELINE
# ============================================================

def save_and_reload_pipeline(
    best_pipeline,
    X_test,
    y_test
):

    print("\n" + "=" * 80)
    print("TASK 15 - SAVE AND RELOAD COMPLETE PIPELINE")
    print("=" * 80)

    # --------------------------------------------------------
    # SAVE COMPLETE PIPELINE
    # --------------------------------------------------------

    joblib.dump(
        best_pipeline,
        PIPELINE_PATH
    )

    print(
        f"\nPipeline saved to:"
    )

    print(
        PIPELINE_PATH
    )

    # --------------------------------------------------------
    # RELOAD
    # --------------------------------------------------------

    loaded_pipeline = joblib.load(
        PIPELINE_PATH
    )

    print(
        "\nPipeline successfully reloaded."
    )

    # --------------------------------------------------------
    # PREDICT RAW INPUT
    # --------------------------------------------------------

    predictions = (
        loaded_pipeline
        .predict(X_test)
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        "\nReloaded pipeline accuracy:"
    )

    print(
        f"{accuracy:.4f}"
    )

    print(
        "\nThe saved artifact contains the "
        "preprocessing and estimator together."
    )

    return loaded_pipeline


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "#" * 80
    )

    print(
        "ZEpto ANALYTICS PIPELINE"
    )

    print(
        "#" * 80
    )

    # --------------------------------------------------------
    # TASK 1
    # --------------------------------------------------------

    df = load_dataset_once()

    missing_percentages = (
        profile_dataset(df)
    )

    # --------------------------------------------------------
    # TASK 2
    # --------------------------------------------------------

    df = clean_dataset(
        df,
        missing_percentages
    )

    # --------------------------------------------------------
    # TASK 3
    # --------------------------------------------------------

    univariate_analysis(
        df
    )

    # --------------------------------------------------------
    # TASK 4
    # --------------------------------------------------------

    bivariate_analysis(
        df
    )

    # --------------------------------------------------------
    # TASK 5
    # --------------------------------------------------------

    multivariate_analysis(
        df
    )

    # --------------------------------------------------------
    # TASK 6
    # --------------------------------------------------------

    standardization_check(
        df
    )

    # --------------------------------------------------------
    # TASK 7
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        df
    )

    # --------------------------------------------------------
    # TASK 9/10
    # --------------------------------------------------------

    (
        trained_models,
        predictions,
        probabilities,
        classification_metrics_df
    ) = train_classifiers(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # --------------------------------------------------------
    # DECISION TREE
    # --------------------------------------------------------

    visualize_decision_tree(
        trained_models
    )

    # --------------------------------------------------------
    # TASK 11
    # --------------------------------------------------------

    imbalance_results = (
        imbalance_comparison(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    imbalance_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "imbalance_comparison.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # TASK 12
    # --------------------------------------------------------

    (
        tuned_rf_pipeline,
        grid_search
    ) = random_forest_tuning(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # --------------------------------------------------------
    # TASK 13
    # --------------------------------------------------------

    (
        regression_pipeline,
        regression_metrics
    ) = regression_task(
        df
    )

    # --------------------------------------------------------
    # TASK 14
    # --------------------------------------------------------

    best_classifier = (
        model_comparison(
            classification_metrics_df,
            regression_metrics
        )
    )

    # --------------------------------------------------------
    # TASK 15
    # --------------------------------------------------------

    # Save the best classifier according to F1.
    best_classifier_name = (
        best_classifier["model"]
    )

    best_pipeline = (
        trained_models[
            best_classifier_name
        ]
    )

    save_and_reload_pipeline(
        best_pipeline,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print(
        "\n"
        + "#" * 80
    )

    print(
        "ANALYTICS PIPELINE COMPLETED"
    )

    print(
        "#" * 80
    )

    print(
        "\nGenerated:"
    )

    print(
        f"- {CSV_PATH}"
    )

    print(
        f"- {OUTPUT_DIR}"
    )

    print(
        f"- {PIPELINE_PATH}"
    )


if __name__ == "__main__":

    main()
