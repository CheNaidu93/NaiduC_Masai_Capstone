import os
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PIPELINE_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "best_pipeline.joblib"
)


# Raw new passenger data.
# No preprocessing is performed here.

new_passenger = pd.DataFrame({
    "pclass": [1],
    "sex": ["female"],
    "age": [30],
    "sibsp": [0],
    "parch": [0],
    "fare": [100.0],
    "embarked": ["C"]
})


pipeline = joblib.load(
    PIPELINE_PATH
)


prediction = pipeline.predict(
    new_passenger
)

probability = pipeline.predict_proba(
    new_passenger
)[:, 1]


print(
    "Prediction:",
    prediction[0]
)

print(
    "Survival probability:",
    probability[0]
)
