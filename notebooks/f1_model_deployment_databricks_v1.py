# Databricks notebook source
# MAGIC %md
# MAGIC # Homework #5: F1 Model Deployment
# MAGIC
# MAGIC This notebook trains two F1 podium-finish prediction models, logs both models in MLflow, and writes each model's holdout predictions into a separate table in my Databricks database.

# COMMAND ----------

from datetime import datetime, timezone
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC `CATALOG_NAME` and `DATABASE_NAME` are set to my assigned Unity Catalog location for the assignment. The path `/Volumes/gr5069/rl3592/...` shows that my personal database/schema is `rl3592` in the `gr5069` catalog.

# COMMAND ----------

CATALOG_NAME = "gr5069"
DATABASE_NAME = "rl3592"
VOLUME_PATH = "/Volumes/gr5069/raw/f1_data"
DATASET_PATH = f"{VOLUME_PATH}/results.csv"

try:
    CURRENT_USER = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .userName()
        .get()
    )
    EXPERIMENT_NAME = f"/Users/{CURRENT_USER}/f1-homework-5-model-deployment"
except Exception:
    EXPERIMENT_NAME = "f1-homework-5-model-deployment"

RF_TABLE_NAME = "f1_random_forest_podium_predictions"
GB_TABLE_NAME = "f1_gradient_boosting_podium_predictions"

TARGET_COLUMN = "podium_finish"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# COMMAND ----------

def qualified_table(table_name):
    return f"`{CATALOG_NAME}`.`{DATABASE_NAME}`.`{table_name}`"


def table_path(table_name):
    return f"{CATALOG_NAME}.{DATABASE_NAME}.{table_name}"


spark.sql(f"USE CATALOG `{CATALOG_NAME}`")
spark.sql(f"USE SCHEMA `{DATABASE_NAME}`")

for table_name in [RF_TABLE_NAME, GB_TABLE_NAME]:
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {qualified_table(table_name)} (
            resultId BIGINT,
            raceId BIGINT,
            driverId BIGINT,
            constructorId BIGINT,
            run_id STRING,
            model_name STRING,
            actual_podium_finish INT,
            predicted_podium_finish INT,
            podium_probability DOUBLE,
            prediction_created_at TIMESTAMP
        )
        USING DELTA
        """
    )

display(spark.sql(f"SHOW TABLES IN `{CATALOG_NAME}`.`{DATABASE_NAME}`"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load F1 Results Data
# MAGIC
# MAGIC The target is `podium_finish`, defined as 1 when `positionOrder <= 3` and 0 otherwise. I use only pre-result identifiers and grid position as baseline features so the model does not directly learn from final race outcome columns like points or finishing position.

# COMMAND ----------

df_f1_spark = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(DATASET_PATH)
)

display(df_f1_spark)

# COMMAND ----------

df = df_f1_spark.toPandas()

if "positionOrder" not in df.columns:
    raise ValueError("This notebook expects results.csv to include positionOrder.")

df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
df = df.dropna(subset=["positionOrder"]).copy()
df[TARGET_COLUMN] = np.where(df["positionOrder"] <= 3, 1, 0)

candidate_feature_columns = [
    "grid",
    "driverId",
    "constructorId",
    "raceId",
    "number",
]
feature_columns = [column for column in candidate_feature_columns if column in df.columns]

if not feature_columns:
    raise ValueError("No expected feature columns were found in results.csv.")

identifier_columns = ["resultId", "raceId", "driverId", "constructorId"]
missing_identifier_columns = [
    column for column in identifier_columns if column not in df.columns
]

if missing_identifier_columns:
    raise ValueError(
        f"results.csv is missing required identifier columns: {missing_identifier_columns}"
    )

selected_columns = list(dict.fromkeys(identifier_columns + feature_columns + [TARGET_COLUMN]))
model_df = df[selected_columns].copy()

for column in feature_columns + identifier_columns:
    model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

model_df = model_df.dropna(subset=identifier_columns + [TARGET_COLUMN]).copy()
model_df[feature_columns] = model_df[feature_columns].fillna(-1)
model_df[TARGET_COLUMN] = model_df[TARGET_COLUMN].astype(int)

for column in identifier_columns:
    model_df[column] = model_df[column].astype("int64")

display(model_df.head())
display(model_df[TARGET_COLUMN].value_counts().rename_axis(TARGET_COLUMN).reset_index(name="count"))

# COMMAND ----------

X = model_df[feature_columns]
y = model_df[TARGET_COLUMN]
prediction_context = model_df[identifier_columns]

X_train, X_test, y_train, y_test, context_train, context_test = train_test_split(
    X,
    y,
    prediction_context,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training rows: {len(X_train):,}")
print(f"Test rows: {len(X_test):,}")
print(f"Feature columns: {feature_columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train, Track, and Store Predictions
# MAGIC
# MAGIC Each MLflow run logs hyperparameters, the fitted model, at least four metrics, and multiple artifacts. After each run finishes, the notebook writes that model's holdout predictions into its matching Delta table.

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT_NAME)

model_specs = [
    {
        "model_name": "random_forest",
        "table_name": RF_TABLE_NAME,
        "estimator": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hyperparameters": {
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "random_state": RANDOM_STATE,
        },
    },
    {
        "model_name": "gradient_boosting",
        "table_name": GB_TABLE_NAME,
        "estimator": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        "hyperparameters": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_samples_leaf": 2,
            "random_state": RANDOM_STATE,
        },
    },
]


def evaluate_classifier(y_true, predictions, probabilities):
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, probabilities)
    except ValueError:
        metrics["roc_auc"] = np.nan

    return metrics


def build_prediction_output(context, y_true, predictions, probabilities, run_id, model_name):
    output = context.reset_index(drop=True).copy()
    output["run_id"] = run_id
    output["model_name"] = model_name
    output["actual_podium_finish"] = y_true.reset_index(drop=True).astype(int)
    output["predicted_podium_finish"] = pd.Series(predictions).astype(int)
    output["podium_probability"] = pd.Series(probabilities).astype(float)
    output["prediction_created_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    return output[
        [
            "resultId",
            "raceId",
            "driverId",
            "constructorId",
            "run_id",
            "model_name",
            "actual_podium_finish",
            "predicted_podium_finish",
            "podium_probability",
            "prediction_created_at",
        ]
    ]


def log_artifacts(model, predictions_output, y_true, predictions, model_name):
    with tempfile.TemporaryDirectory(prefix=f"{model_name}-artifacts-") as temp_dir:
        temp_dir_path = Path(temp_dir)

        predictions_output.to_csv(temp_dir_path / "predictions.csv", index=False)

        report_df = pd.DataFrame(
            classification_report(y_true, predictions, output_dict=True, zero_division=0)
        ).transpose()
        report_df.to_csv(temp_dir_path / "classification_report.csv")

        if hasattr(model, "feature_importances_"):
            importance_df = (
                pd.DataFrame(
                    {
                        "feature": feature_columns,
                        "importance": model.feature_importances_,
                    }
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
            importance_df.to_csv(temp_dir_path / "feature_importance.csv", index=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            importance_df.sort_values("importance", ascending=True).plot.barh(
                x="feature", y="importance", ax=ax, legend=False
            )
            ax.set_title(f"{model_name} Feature Importance")
            ax.set_xlabel("Importance")
            fig.tight_layout()
            fig.savefig(temp_dir_path / "feature_importance.png", bbox_inches="tight", dpi=200)
            display(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_true, predictions, ax=ax)
        ax.set_title(f"{model_name} Confusion Matrix")
        fig.tight_layout()
        fig.savefig(temp_dir_path / "confusion_matrix.png", bbox_inches="tight", dpi=200)
        display(fig)
        plt.close(fig)

        mlflow.log_artifacts(temp_dir, artifact_path="artifacts")


from pyspark.sql.functions import col

def save_predictions_to_table(predictions_output, table_name):
    spark_predictions = spark.createDataFrame(predictions_output)

    spark_predictions = (
        spark_predictions
        .withColumn("resultId", col("resultId").cast("bigint"))
        .withColumn("raceId", col("raceId").cast("bigint"))
        .withColumn("driverId", col("driverId").cast("bigint"))
        .withColumn("constructorId", col("constructorId").cast("bigint"))
        .withColumn("run_id", col("run_id").cast("string"))
        .withColumn("model_name", col("model_name").cast("string"))
        .withColumn("actual_podium_finish", col("actual_podium_finish").cast("int"))
        .withColumn("predicted_podium_finish", col("predicted_podium_finish").cast("int"))
        .withColumn("podium_probability", col("podium_probability").cast("double"))
        .withColumn("prediction_created_at", col("prediction_created_at").cast("timestamp"))
    )

    spark.sql(f"DELETE FROM {qualified_table(table_name)}")
    spark_predictions.write.mode("append").format("delta").saveAsTable(
        table_path(table_name)
    )



run_results = []

for spec in model_specs:
    model_name = spec["model_name"]
    model = spec["estimator"]

    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probability_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(X_test)[:, probability_index]

        metrics = evaluate_classifier(y_test, predictions, probabilities)

        mlflow.log_params(spec["hyperparameters"])
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset_path", DATASET_PATH)
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("feature_columns", ", ".join(feature_columns))
        mlflow.log_param("prediction_table", table_path(spec["table_name"]))
        mlflow.log_metrics({key: value for key, value in metrics.items() if pd.notna(value)})

        signature = infer_signature(X_train.head(5), model.predict(X_train.head(5)))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
        )

        predictions_output = build_prediction_output(
            context_test,
            y_test,
            predictions,
            probabilities,
            run.info.run_id,
            model_name,
        )

        log_artifacts(model, predictions_output, y_test, predictions, model_name)
        save_predictions_to_table(predictions_output, spec["table_name"])

        run_record = {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "prediction_table": table_path(spec["table_name"]),
            **metrics,
        }
        run_results.append(run_record)

        print(f"Finished {model_name}")
        print(f"  run_id: {run.info.run_id}")
        print(f"  f1: {metrics['f1']:.4f}")
        print(f"  prediction table: {table_path(spec['table_name'])}")

# COMMAND ----------

leaderboard = pd.DataFrame(run_results).sort_values("f1", ascending=False).reset_index(drop=True)
display(leaderboard)

best_run = leaderboard.iloc[0]
print(
    "Best model: {model_name} with F1={f1:.4f}. "
    "I selected the best run by F1 because this is a binary classification problem "
    "and F1 balances precision and recall for podium predictions.".format(**best_run)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Database Tables
# MAGIC
# MAGIC These cells show that both prediction tables exist and contain model predictions.

# COMMAND ----------

display(spark.sql(f"SELECT COUNT(*) AS prediction_rows FROM {qualified_table(RF_TABLE_NAME)}"))
display(spark.sql(f"SELECT COUNT(*) AS prediction_rows FROM {qualified_table(GB_TABLE_NAME)}"))

display(spark.table(table_path(RF_TABLE_NAME)).limit(20))
display(spark.table(table_path(GB_TABLE_NAME)).limit(20))
