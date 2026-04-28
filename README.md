# Homework #5: Model Deployment

This repo contains a Databricks notebook for the F1 model-deployment assignment.

## Assignment Prompt

Using the F1 dataset, build predictive models, log them in MLflow, and write the model predictions into a database.

1. Create two new tables in your own database to store predictions from each model.
2. Build two predictive models with MLflow, logging hyperparameters, the model itself, four metrics, and two artifacts.
3. Store each model's predictions in the corresponding table in your own database.
4. Push the completed code to GitHub.

## Submission Files

- `notebooks/f1_model_deployment_databricks.py`
  - Databricks notebook source file.
  - Creates the prediction tables.
  - Trains a Random Forest and Gradient Boosting classifier.
  - Logs both runs to MLflow.
  - Writes holdout predictions into two Delta tables.
- `requirements.txt`
  - Local package list for the Python modeling dependencies.

## Modeling Choice

The notebook predicts whether a driver finishes on the podium.

- Target: `podium_finish`
- Definition: `1` when `positionOrder <= 3`, otherwise `0`
- Source data: `/Volumes/gr5069/raw/f1_data/results.csv`
- Baseline features: `grid`, `driverId`, `constructorId`, `raceId`, and `number`

I used two classifiers:

- Random Forest
- Gradient Boosting

Both models log hyperparameters, the fitted model, and the following metrics:

- accuracy
- balanced accuracy
- precision
- recall
- F1
- ROC AUC

Both runs also log multiple artifacts:

- predictions CSV
- classification report CSV
- feature importance CSV
- feature importance plot
- confusion matrix plot

## Database Tables

The notebook creates and writes to these tables in my database:

- `michaelliruoxi.f1_random_forest_podium_predictions`
- `michaelliruoxi.f1_gradient_boosting_podium_predictions`

If the Databricks workspace uses a different assigned database name, update `DATABASE_NAME` at the top of the notebook before running.

## How To Run

1. Import `notebooks/f1_model_deployment_databricks.py` into Databricks.
2. Confirm `DATABASE_NAME = "michaelliruoxi"` matches the assigned database/schema.
3. Run the notebook from top to bottom.
4. Open the MLflow experiment named `/Users/<your-databricks-user>/f1-homework-5-model-deployment`.
5. Take screenshots of the MLflow experiment/runs and the notebook cells showing the prediction tables.

## Best Model Explanation

The notebook displays a leaderboard sorted by F1 score. I select the best model by F1 because this is a binary classification problem and F1 balances precision and recall for podium predictions.
