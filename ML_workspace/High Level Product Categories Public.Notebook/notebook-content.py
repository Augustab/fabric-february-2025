# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "029e2656-ce72-41b7-bfa8-f0b8a6cd2917",
# META       "default_lakehouse_name": "Gold",
# META       "default_lakehouse_workspace_id": "d1bc3c0e-7995-4f9b-b79b-b096c69fd5d9"
# META     }
# META   }
# META }

# CELL ********************

%run env_variables

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pyspark.sql.functions as F
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from synapse.ml.services.openai import OpenAIEmbedding
from sklearn.decomposition import PCA
from synapse.ml.services.openai import OpenAIChatCompletion
from pyspark.sql import Row

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = spark.sql("SELECT product_category_name_english FROM Gold.products")

df = df.select("product_category_name_english").distinct()
df = df.filter(F.col("product_category_name_english").isNotNull())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

deployment_name_embeddings = "text-embedding-ada-002"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark",
# META   "frozen": false,
# META   "editable": true
# META }

# MARKDOWN ********************

# ![image-alt-text](https://images.ctfassets.net/k07f0awoib97/2n4uIQh2bAX7fRmx4AGzyY/a1bc6fa1e2d14ff247716b5f589a2099/Screen_Recording_2023-06-03_at_4.52.54_PM.gif)

# CELL ********************

df = df.withColumn("product_category_name_english", F.regexp_replace(df["product_category_name_english"],"_", " "))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

embedding = (
    OpenAIEmbedding()
    .setSubscriptionKey(key)
    .setDeploymentName(deployment_name_embeddings)
    .setCustomServiceName(service_name)
    .setDeploymentName("text-embedding-ada-002")
    .setTextCol("product_category_name_english")
    .setErrorCol("error")
    .setOutputCol("embeddings")
)

completed_df = embedding.transform(df).cache()
display(completed_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

collected = list(completed_df.collect())
matrix = np.array([[r["embeddings"]] for r in collected])[:, 0, :].astype(np.float64)
print(matrix.shape)

# Perform PCA to reduce dimensionality to keep 95% of the internal variance
pca = PCA(n_components=0.95, random_state=42)
vis_dims = pca.fit_transform(matrix)

print(vis_dims.shape)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

num = 18

kmeans = KMeans(n_clusters=num, random_state=42)
kmeans.fit(vis_dims)

# Get cluster labels
labels = kmeans.labels_

# Initialize a dictionary to store the groups
grouped_categories = {f'Group_{i+1}': [] for i in range(num)}

product_categories = df.select("product_category_name_english").collect()

# Assign each product category to the respective cluster group
for i, label in enumerate(labels):
    category = product_categories[i][0]
    grouped_categories[f'Group_{label+1}'].append(category)

# Convert the dictionary to a DataFrame for better visualization
grouped_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in grouped_categories.items()]))

print(grouped_df)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Previously
# deployment_name = "gpt-4"

deployment_name = "gpt-4o-2024-08-06"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark",
# META   "frozen": false,
# META   "editable": true
# META }

# CELL ********************

system_message = "You are an AI assistant that helps a business create high level product categories."

# Function to create message Rows
def make_message(role, content):
    return Row(role=role, content=content, name=role)

# Function to create a prompt for each group of product categories
def create_chat_row(group_name, categories):
    if len(categories) == 1:
        # If only one category, keep the original name
        return Row(messages=[], collective_name=categories[0])
    else:
        # If multiple categories, generate a prompt
        user_message = (
            f"I will give you a group of two or more product categories. "
            f"Your task is to come up with a suitable collective name that best represents all of the categories I give you. "
            f"You should respond with only your suggestion. Include nothing else. "
            f"Remember to only respond with your suggestion. Product categories= {', '.join(categories)}"
        )
        return Row(messages=[
            make_message("system", system_message),
            make_message("user", user_message)
        ], collective_name=None)


# Create a list of chat messages for each group
chat_rows = []
for group_name, categories in grouped_df.items():
    chat_rows.append(create_chat_row(group_name, categories.dropna().tolist()))

# Convert the list of messages to a Spark DataFrame
chat_messages_df = spark.createDataFrame(chat_rows)

# Filter out rows with empty messages to avoid sending them to OpenAI
non_empty_prompts_df = chat_messages_df.filter("size(messages) > 0")


display(non_empty_prompts_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark",
# META   "frozen": false,
# META   "editable": true
# META }

# CELL ********************

# Create the OpenAIChatCompletion object
chat_completion = (
    OpenAIChatCompletion()
    # .setSubscriptionKey(key)
    .setDeploymentName(deployment_name)
    # .setCustomServiceName(service_name)
    .setMessagesCol("messages")
    .setErrorCol("error")
    .setOutputCol("chat_completions")
)

# Run the ChatCompletion transformation only on non-empty prompts
result_df = chat_completion.transform(non_empty_prompts_df)

display(result_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(final_spark_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
