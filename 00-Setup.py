# Databricks notebook source
def display_slide(slide_id, slide_number):
  displayHTML(f'''
  <div style="width:1150px; margin:auto">
  <iframe
    src="https://docs.google.com/presentation/d/{slide_id}/embed?slide={slide_number}"
    frameborder="0"
    width="1150"
    height="600"
  ></iframe></div>
  ''')

# COMMAND ----------

CALHOUSING_DATA_PATH = '/Users/sriharsha.jana@databricks.com/daiwt_mumbai/calhousing'

# COMMAND ----------

from sklearn.datasets import fetch_california_housing
import pickle
import pandas as pd

reset_data = False
if reset_data:
  housing = fetch_california_housing()
  housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
  housing_df[housing.target_names[0]] = housing.target
  spark.createDataFrame(housing_df).write.mode('overwrite').format('delta').save(CALHOUSING_DATA_PATH)
