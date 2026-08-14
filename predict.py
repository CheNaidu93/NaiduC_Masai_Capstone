from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "artifacts" / "best_pipeline.joblib"

pipeline = joblib.load(MODEL)

# Raw/unprocessed input: the pipeline performs imputation,
# encoding and scaling internally.
new_passenger = pd.DataFrame(
    [
        {
            "Pclass": 3,
            "Sex": "male",
            "Age": 30,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 8.05,
            "Embarked": "S",
        }
    ]
)

prediction = pipeline.predict(new_passenger)

print("Raw input:")
print(new_passenger)
print("\nReloaded pipeline prediction:")
print(prediction)
