from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn
import pandas as pd
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("diabetes_experiment")

def modelling(
    df,
    target_col,
    model,
    param_grid,
    test_size=0.2,
    cv=5
):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    with mlflow.start_run(run_name="hyperparameter_tuning"):

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_param("model_name", best_model.__class__.__name__)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("cv", cv)
        mlflow.log_param("param_grid", json.dumps(param_grid))

        for param, value in grid.best_params_.items():
            mlflow.log_param(param, value)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_pos", report["1"]["precision"])
        mlflow.log_metric("recall_pos", report["1"]["recall"])
        mlflow.log_metric("f1_pos", report["1"]["f1-score"])

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            pip_requirements=[
                "scikit-learn",
                "pandas",
                "numpy",
                "matplotlib",
                "seaborn",
                "mlflow"
            ]
        )

        metric_info = {
            "accuracy": acc,
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1_score": report["1"]["f1-score"]
        }

        metric_path = BASE_DIR / "metric_info.json"
        with open(metric_path, "w") as f:
            json.dump(metric_info, f, indent=4)

        mlflow.log_artifact(str(metric_path))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Training Confusion Matrix")
        plt.tight_layout()

        cm_path = BASE_DIR / "training_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(str(cm_path))

        estimator_path = BASE_DIR / "estimator.html"
        with open(estimator_path, "w") as f:
            f.write("<pre>")
            f.write(str(best_model))
            f.write("</pre>")

        mlflow.log_artifact(str(estimator_path))

    return {
        "best_model": best_model,
        "accuracy": acc,
        "best_params": grid.best_params_
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MLflow GridSearch")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--target_col", type=str, default="Outcome")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=5)

    parser.add_argument(
        "--param_grid",
        type=str,
        required=True,
        help="JSON string for GridSearchCV param_grid"
    )

    args = parser.parse_args()

    df = pd.read_csv(BASE_DIR / args.dataset)

    param_grid = json.loads(args.param_grid)

    model = RandomForestClassifier(random_state=42)

    result = modelling(
        df=df,
        target_col=args.target_col,
        model=model,
        param_grid=param_grid,
        test_size=args.test_size,
        cv=args.cv
    )

    print("Best accuracy:", result["accuracy"])
    print("Best params:", result["best_params"])
