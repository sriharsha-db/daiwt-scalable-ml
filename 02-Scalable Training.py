# Databricks notebook source
# MAGIC %run ./00-Setup

# COMMAND ----------

display_slide('1hn4e_2HNrwNXly2xDF6YhwjrVK3lzgJO', 8)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regression with XGBoost and MLlib pipelines
# MAGIC
# MAGIC For more information about the PySpark ML SparkXGBRegressor estimator used in this notebook, see [Xgboost SparkXGBRegressor API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.spark.SparkXGBRegressor).
# MAGIC
# MAGIC #### Requirements
# MAGIC Databricks Runtime for Machine Learning 12.0 ML or above.

# COMMAND ----------

df = spark.read.csv("/databricks-datasets/bikeSharing/data-001/hour.csv", header="true", inferSchema="true")
df = df.drop("instant").drop("dteday").drop("casual").drop("registered")
df.cache()

# COMMAND ----------

train, test = df.randomSplit([0.7, 0.3], seed = 0)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, VectorIndexer

# Remove the target column from the input feature set.
featuresCols = df.columns
featuresCols.remove('cnt')

# vectorAssembler combines all feature columns into a single feature vector column, "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")

# vectorIndexer identifies categorical features and indexes them, and creates a new column "features". 
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)

# COMMAND ----------

from xgboost.spark import SparkXGBRegressor
xgb_regressor = SparkXGBRegressor(num_workers=2, label_col="cnt", missing=0.0)

# COMMAND ----------

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, xgb_regressor])
pipelineModel = pipeline.fit(train)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Distributed training on a PyTorch file
# MAGIC
# MAGIC Distributed training on PyTorch is often done by creating a file (`train.py`) and using the `torchrun` CLI to run distributed training using that file. Databricks streamlines that process by allowing you to import a file (or even a repository) and use a Databricks notebook to start distributed training on that file using the TorchDistributor API. The example file that is used in this example is: `/Workspace/Repos/user.name@databricks.com/.../Basic_MNIST/train.py`.
# MAGIC
# MAGIC This file is laid out similar to other solutions that use `torchrun` under the hood for distributed training.
# MAGIC
# MAGIC #### Requirements
# MAGIC - Databricks Runtime ML 13.0 and above
# MAGIC - (Recommended) GPU instances

# COMMAND ----------

import torch

# COMMAND ----------

USE_GPU = torch.cuda.is_available()
NUM_GPUS_PER_NODE = 1
NUM_PROCESSES = NUM_GPUS_PER_NODE*2
username = spark.sql("SELECT current_user()").first()['current_user()']
repo_path = f'/Workspace/Repos/{username}/daiwt-example/pytorch_script/mnist_train.py'

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

TorchDistributor(num_processes=NUM_PROCESSES,
                 local_mode=False, 
                 use_gpu=USE_GPU)\
          .run(repo_path, 
               "--batch-size=256", 
               "--test-batch-size=128", 
               "--epochs=1", 
               "--lr=1e-3",
               "--gamma=0.7",
               "--seed=123",
               "--log-interval=50")
