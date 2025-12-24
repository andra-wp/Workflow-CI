from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("diabetes_al_experiment")


def train_with_mlflow(df, target_col, model):
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="rf_training"):

        X = df.drop(columns=[target_col])
        y = df[target_col].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

    return model, acc


def main():
    BASE_DIR = Path(__file__).resolve().parent
    DATASET_PATH = BASE_DIR / "diabetes_preprocessing.csv"

    df = pd.read_csv(DATASET_PATH)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    _, acc = train_with_mlflow(
        df=df,
        target_col="Outcome",
        model=model
    )

    print("Accuracy:", acc)


if __name__ == "__main__":
    main()
