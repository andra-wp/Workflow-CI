from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def train_with_mlflow(df, target_col, model):
    mlflow.sklearn.autolog()

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

    input_example = X_train.iloc[:5]

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=input_example
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", acc)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    report = classification_report(y_test, y_pred)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    return model, acc, cm


def main():
    df = pd.read_csv("diabetes_preprocessing.csv")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    _, acc, _ = train_with_mlflow(df, "Outcome", model)
    print("Accuracy:", acc)


if __name__ == "__main__":
    main()
