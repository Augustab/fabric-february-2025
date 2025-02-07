# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "3aa131c8-68b8-4ee8-a79c-e785d64e1c88",
# META       "default_lakehouse_name": "Raw",
# META       "default_lakehouse_workspace_id": "082d6063-1af7-4f12-931b-ec064fcbed1f"
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Setup notebook
# 1. Get the relevant capacity Id
# 2. Creates the required workspaces
# 2. Creates the initial lakehouses
# 3. Install kaggle
# 4. Manually attach the Raw-lakehouse to have a valid path to download to
# 5. Download and unzip Olist E-commerce dataset
# 7. Load the CSV-files as Delta tables into Raw-lakehouse

# CELL ********************

from sempy import fabric
client = fabric.FabricRestClient()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 1. Get Capacity Id

# CELL ********************

capacities = client.get("v1/capacities").json()
trial_capacity_id = [capacity["id"] for capacity in capacities["value"] if 'Trial' in capacity["displayName"]][0]
trial_capacity_id

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 2. Setup workspaces

# CELL ********************

ws_raw_name= "Olist_Raw"
ws_ml_name= "Olist_ML"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Create Fabric workspaces using trial capacity
raw_ws = fabric.create_workspace(ws_raw_name, capacity_id=trial_capacity_id)
ml_ws = fabric.create_workspace(ws_ml_name, capacity_id=trial_capacity_id)
raw_ws, ml_ws

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

raw_ws = fabric.resolve_workspace_id(ws_raw_name)
ml_ws = fabric.resolve_workspace_id(ws_ml_name)
raw_ws, ml_ws

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 3. Setup Lakehouses

# CELL ********************

# Setup lakehouses for initial raw lakehouse and the eventual Semantic Link Lake
fabric.create_lakehouse("Raw", workspace=raw_ws)
fabric.create_lakehouse("Gold", workspace=ml_ws)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 4. Install and setup authentication for Kaggle API

# CELL ********************

!pip install kaggle -q

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import os
# Required to import Kaggle and use the API
os.environ['KAGGLE_USERNAME'] = "YOUR KAGGLE USERNAME"
os.environ['KAGGLE_KEY'] = "YOUR KAGGLE KEY"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ###  5. Manually Add/attach the "Raw"-lakehouse

# MARKDOWN ********************

# ### 6. Download Olist e-commerce Dataset

# CELL ********************

import kaggle
raw_lakehouse = mssparkutils.lakehouse.get("Raw", workspaceId=raw_ws)
kaggle.api.dataset_download_files(dataset = "olistbr/brazilian-ecommerce/", 
                                path='/lakehouse/default/Files/olist_raw', #f{raw_lakehouse["properties"]["abfsPath"]}/Files/olist_raw', 
                                force=False,
                                unzip=True)
kaggle.api.dataset_download_files(dataset = "olistbr/marketing-funnel-olist/", 
                                path='/lakehouse/default/Files/olist_marketing_raw', #f{raw_lakehouse["properties"]["abfsPath"]}/Files/olist_raw', 
                                force=False,
                                unzip=True)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 7. Load the CSV-files as Delta tables into `Raw`-Lakehouse

# CELL ********************

# Convert to delta tables
raw_lakehouse_path = raw_lakehouse["properties"]["abfsPath"]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

files = notebookutils.fs.ls(f'{raw_lakehouse_path}/Files/olist_raw')
for file in files:
    print("Converting", file.name, "to Delta")
    table_name = "_".join(file.name.split("_")[1:-1])
    (spark
        .read
        .csv(file.path, header=True)
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(f"{raw_lakehouse_path}/Tables/{table_name}")
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

files = notebookutils.fs.ls(f'{raw_lakehouse_path}/Files/olist_marketing_raw')
for file in files:
    print("Converting", file.name, "to Delta")
    table_name = "_".join(file.name.split("_")[1:-1])
    (spark
        .read
        .csv(file.path, header=True)
        .write
        .format("delta")
        .option("overwriteSchema", "true")
        .mode("overwrite")
        .save(f"{raw_lakehouse_path}/Tables/{table_name}")
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Setup Semantic Model
# 1. Enable REad/write
# 1. Install Semantic link labs
# 2. Create empty Semantic Model
# 3. Sync from Lakehouse to Semantic model 

# CELL ********************

!pip install semantic-link-labs -q

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Create empty Semantic Model

# CELL ********************

import sempy_labs as labs
from sempy_labs import tom
sales_model = "Sales"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Create initial empty semantic model
labs.create_blank_semantic_model(sales_model, workspace=raw_ws)
labs.directlake.update_direct_lake_model_lakehouse_connection(sales_model, ws_raw_name, raw_lakehouse.displayName)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Add all tables to the semantic model
with labs.tom.connect_semantic_model(sales_model, workspace=raw_ws) as tom:
    for table in spark.catalog.listTables():
        tom.add_table(table.name)
        tom.add_entity_partition(table.name, table.name)

    
    #labs.directlake.add_table_to_direct_lake_semantic_model("Sales", table.name, table.name, workspace=ws_raw_name)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
