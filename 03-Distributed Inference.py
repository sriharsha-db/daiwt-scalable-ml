# Databricks notebook source
# MAGIC %run ./00-Setup

# COMMAND ----------

display_slide('1hn4e_2HNrwNXly2xDF6YhwjrVK3lzgJO', 13)

# COMMAND ----------

from pyspark.sql.functions import collect_list, concat_ws, col, count, pandas_udf, udf
from pyspark.sql.types import *
from transformers import pipeline
import pandas as pd
import mlflow
import re

# COMMAND ----------

DATA_PATH = "/Volumes/uc_sriharsha_jana/test_db/shj_daiwt/"
models_cache_dir = "/dbfs/Users/sriharsha.jana@databricks.com/llm"
display(spark.read.parquet(DATA_PATH))

# COMMAND ----------

remove_regex = re.compile(r"(&[#0-9]+;|<[^>]+>|\[\[[^\]]+\]\]|[\r\n]+)")
split_regex = re.compile(r"([?!.]\s+)")

def clean_text(text, max_tokens):
  if not text:
    return ""
  text = remove_regex.sub(" ", text.strip()).strip()
  approx_tokens = 0
  cleaned = ""
  for fragment in split_regex.split(text):
    approx_tokens += len(fragment.split(" "))
    if (approx_tokens > max_tokens):
      break
    cleaned += fragment
  return cleaned.strip()

@udf('string')
def clean_review_udf(review):
  return clean_text(review, 100)

@udf('string')
def clean_summary_udf(summary):
  return clean_text(summary, 20)

# COMMAND ----------

spark.read.parquet(DATA_PATH).select("product_id", "review_body", "review_headline").\
  sample(0.1, seed=42).\
  withColumn("review_body", clean_review_udf("review_body")).\
  withColumn("review_headline", clean_summary_udf("review_headline")).\
  filter("LENGTH(review_body) > 0 AND LENGTH(review_headline) > 0").\
  write.format("delta").saveAsTable("uc_sriharsha_jana.test_db.amzcam_review_cleaned")

# COMMAND ----------

model_name = "blrdaiwt_summarization"
model_uri = f"models:/{model_name}/3"
# create spark user-defined function for model prediction
t5_summarizer_udf = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="string")

# COMMAND ----------

camera_reviews_df = spark.read.table("uc_sriharsha_jana.test_db.amzcam_review_cleaned").sample(0.1, False)

review_by_product_df = camera_reviews_df.groupBy("product_id").\
  agg(collect_list("review_body").alias("review_array"), count("*").alias("n")).\
  filter("n >= 10").\
  select("product_id", "n", concat_ws(" ", col("review_array")).alias("reviews"))

# COMMAND ----------

review_by_product_df.withColumn("summary", t5_summarizer_udf("reviews")).\
  write.mode("overwrite").format("delta").saveAsTable("uc_sriharsha_jana.test_db.amzcam_review_summarised")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from uc_sriharsha_jana.test_db.amzcam_review_summarised limit 4;
