# Databricks notebook source
# MAGIC %run ./00-Setup

# COMMAND ----------

display_slide('1hn4e_2HNrwNXly2xDF6YhwjrVK3lzgJO', 1)

# COMMAND ----------

# MAGIC %md 
# MAGIC # MLflow
# MAGIC
# MAGIC <a href="https://mlflow.org/docs/latest/concepts.html" target="_blank">MLflow</a> seeks to address these three core issues:
# MAGIC
# MAGIC * It’s difficult to keep track of experiments
# MAGIC * It’s difficult to reproduce code
# MAGIC * There’s no standard way to package and deploy models
# MAGIC
# MAGIC In the past, when examining a problem, you would have to manually keep track of the many models you created, as well as their associated parameters and metrics. This can quickly become tedious and take up valuable time, which is where MLflow comes in.
# MAGIC
# MAGIC MLflow is pre-installed on the Databricks Runtime for ML.

# COMMAND ----------

import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mlflow.autolog(exclusive=False)

# COMMAND ----------

housing_data = spark.read.format('delta').load(CALHOUSING_DATA_PATH).toPandas()
display(housing_data)

# COMMAND ----------

x_train, x_test, y_train, y_test = train_test_split(housing_data.drop(['MedHouseVal'], axis=1), 
                                                    housing_data[['MedHouseVal']], 
                                                    random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
print(x_train.shape, x_test.shape)      

# COMMAND ----------

# MAGIC %md ## Types of experiments
# MAGIC There are two types of experiments in MLflow: _notebook_ and _workspace_. 
# MAGIC * A notebook experiment is associated with a specific notebook. Databricks creates a notebook experiment by default when a run is started using `mlflow.start_run()` and there is no active experiment.
# MAGIC * Workspace experiments are not associated with any notebook, and any notebook can log a run to these experiments by using the experiment name or the experiment ID when initiating a run. 
# MAGIC
# MAGIC This notebook creates a Random Forest model on a simple dataset and uses the MLflow Tracking API to log the model and selected model parameters and metrics.

# COMMAND ----------

mlflow_experiment_path = f"/Users/sriharsha.jana@databricks.com/daiwt_mlflow_exp"
exp_obj = mlflow.set_experiment(mlflow_experiment_path)
print(exp_obj)

# COMMAND ----------

import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

run_id=None
with mlflow.start_run(run_name="random_forest_run") as run:
  n_estimators = 100
  max_depth = 5
  max_features = 4

  rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf.fit(x_train, y_train)

  target_col = "TARGET"
  pd_test = pd.DataFrame(x_test)
  pd_test[target_col] = y_test
  mlflow.evaluate(
      model=f"runs:/{run.info.run_id}/model",
      data=pd_test,
      targets=target_col,
      model_type="regressor",
      evaluator_config = {"log_model_explainability": True,
                          "explainability_nsamples": 500,
                          "metric_prefix": "test_" , "pos_label": 1}
  )

  pd_train = pd.DataFrame(x_train)
  pd_train[target_col] = y_train
  mlflow.evaluate(
      model=f"runs:/{run.info.run_id}/model",
      data=pd_train,
      targets=target_col,
      model_type="regressor",
      evaluator_config = {"log_model_explainability": False,
                          "metric_prefix": "train_" , "pos_label": 1}
  )

  run_id=run.info.run_id

# COMMAND ----------

exp_data = mlflow.search_runs(experiment_ids=[exp_obj.experiment_id])
display(exp_data)

# COMMAND ----------

display_slide('1hn4e_2HNrwNXly2xDF6YhwjrVK3lzgJO', 6)

# COMMAND ----------

model_uri = "runs:/{run_id}/model".format(run_id=run_id)
model_name = "calhousing_model"
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
print(model_details)

# COMMAND ----------

from mlflow import MlflowClient
mlflow_client = MlflowClient()

# COMMAND ----------

mlflow_client.transition_model_version_stage(name=model_name, 
                                             version=model_details.version, 
                                             stage='Staging',
                                             archive_existing_versions=True)
