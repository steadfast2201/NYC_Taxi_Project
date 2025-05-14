import pathlib
import joblib
import sys
import yaml
import pandas as pd
from sklearn import metrics
from dvclive import Live
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


def evaluate(model, X, y, split, live, save_path):
    """
    Evaluate regression model with RMSE and R².

    Args:
        model: Trained regressor.
        X (pd.DataFrame): Features.
        y (pd.Series): Targets.
        split (str): "train" or "test".
        live (dvclive.Live): DVC Live instance.
        save_path (str): Where to save plots.
    """
    preds = model.predict(X)

    rmse = metrics.root_mean_squared_error(y, preds)
    r2 = metrics.r2_score(y, preds)

    # Log metrics
    live.log_metric(f"rmse_{split}", rmse)
    live.log_metric(f"r2_{split}", r2)
    mlflow.log_metric(f"rmse_{split}", rmse)
    mlflow.log_metric(f"r2_{split}", r2)

    # Predicted vs Actual plot
    fig, ax = plt.subplots()
    ax.scatter(y, preds, alpha=0.4)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "--r", linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted ({split})")
    fig.tight_layout()
    pred_plot_path = f"{save_path}/predicted_vs_actual_{split}.png"
    fig.savefig(pred_plot_path)
    plt.close(fig)
    mlflow.log_artifact(pred_plot_path)

    # Residual plot
    fig, ax = plt.subplots()
    residuals = y - preds
    ax.scatter(preds, residuals, alpha=0.4)
    ax.axhline(y=0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residual Plot ({split})")
    fig.tight_layout()
    res_plot_path = f"{save_path}/residuals_{split}.png"
    fig.savefig(res_plot_path)
    plt.close(fig)
    mlflow.log_artifact(res_plot_path)


def save_importance_plot(live, model, feature_names, save_path):
    """
    Save feature importance plot (for tree-based models).
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names).nlargest(n=10)
    forest_importances.plot.bar(ax=ax)
    ax.set_ylabel("Importance")
    ax.set_title("Top 10 Feature Importances")
    fig.tight_layout()

    importance_path = f"{save_path}/importance.png"
    fig.savefig(importance_path)
    plt.close(fig)

    live.log_image("importance.png", fig)
    mlflow.log_artifact(importance_path)


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    model_file = sys.argv[1]
    input_file = sys.argv[2]

    # Load model
    model = joblib.load(model_file)

    # Load data
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir / "dvclive"
    output_path.mkdir(parents=True, exist_ok=True)

    TARGET = "trip_duration"  # change if your target column is different
    train_df = pd.read_csv(f"{data_path}/train.csv")
    test_df = pd.read_csv(f"{data_path}/test.csv")

    X_train = train_df.drop(columns=TARGET)
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=TARGET)
    y_test = test_df[TARGET]

    feature_names = X_train.columns.tolist()

    mlflow.set_experiment("regression_model_eval")

    with mlflow.start_run(run_name="evaluate_regression_model"):
        with Live(output_path.as_posix(), dvcyaml=False) as live:
            evaluate(model, X_train, y_train, "train", live, output_path.as_posix())
            evaluate(model, X_test, y_test, "test", live, output_path.as_posix())

            # Only if tree-based regressor (like XGB, RandomForest)
            if hasattr(model, "feature_importances_"):
                save_importance_plot(live, model, feature_names, output_path.as_posix())

        # Log model
        signature = infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(
            model, "model", input_example=X_test.head(1), signature=signature
        )


if __name__ == "__main__":
    main()
