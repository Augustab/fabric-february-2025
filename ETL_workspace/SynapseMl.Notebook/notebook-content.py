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

# CELL ********************

import numpy as np 
import pandas as pd 
from scipy import stats 
import os
import matplotlib.pyplot as plt
import seaborn as sns 
import pyspark.sql.functions as F
from synapse.ml.services.translate import Translate

raw_lakehouse_abfsPath = mssparkutils.lakehouse.get("Raw").properties["abfsPath"]
gold_lakehouse_abfsPath = mssparkutils.lakehouse.get("Translated").properties["abfsPath"]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Load all raw files as CSVs
df_item = spark.read.csv(f"{raw_lakehouse_abfsPath}/Files/olist_raw/olist_order_items_dataset.csv", header=True, inferSchema=True)
df_orders = spark.read.csv(f"{raw_lakehouse_abfsPath}/Files/olist_raw/olist_orders_dataset.csv", header=True, inferSchema=True)
# Filter inconsistent date before 2017 and september 2018
# df_item = (df_item.filter(F.year(F.col("order_purchase_timestamp")) >= 2017)
#             .filter(~((F.year(F.col("order_purchase_timestamp")) == 2018)&(F.month(F.col("order_purchase_timestamp"))==9)))
#             )
# Some reviews span mulitple lines/containes new line characters -> Multiline=true
# Also, double quotes in reviews are already escaped as ""
df_reviews = spark.read.csv(f"{raw_lakehouse_abfsPath}/Files/olist_raw/olist_order_reviews_dataset.csv", 
                        header=True,
                        quote="\"",
                        escape="\"",
                        multiLine=True, inferSchema=True)
df_products = spark.read.csv(f"{raw_lakehouse_abfsPath}/Files/olist_raw/olist_products_dataset.csv", header=True, inferSchema=True)
df_products = spark.read.csv(f"{raw_lakehouse_abfsPath}/Files/olist_raw/olist_products_dataset.csv", header=True, inferSchema=True)
df_geolocation = spark.read.csv(f"{raw_lakehouse_abfsPath}/Files/olist_raw/olist_geolocation_dataset.csv", header=True, inferSchema=True)
df_sellers = spark.read.csv(f"{raw_lakehouse_abfsPath}/Files/olist_raw/olist_sellers_dataset.csv", header=True, inferSchema=True)
df_order_pay = spark.read.csv(f"{raw_lakehouse_abfsPath}/Files/olist_raw/olist_order_payments_dataset.csv", header=True, inferSchema=True)
df_customers = spark.read.csv(f"{raw_lakehouse_abfsPath}/Files/olist_raw/olist_customers_dataset.csv", header=True, inferSchema=True)
df_category = spark.read.csv(f"{raw_lakehouse_abfsPath}/Files/olist_raw/product_category_name_translation.csv", header=True, inferSchema=True)
# marketing
df_closed_deals = spark.read.csv(f"{raw_lakehouse_abfsPath}/Files/olist_marketing_raw/olist_closed_deals_dataset.csv", header=True, inferSchema=True)
df_qualified_leads = spark.read.csv(f"{raw_lakehouse_abfsPath}/Files/olist_marketing_raw/olist_marketing_qualified_leads_dataset.csv", header=True, inferSchema=True)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Save tables which does not require translation
df_item.write.save(f"{gold_lakehouse_abfsPath }/Tables/order_items", format="delta", mode="overwrite")
df_orders.write.save(f"{gold_lakehouse_abfsPath }/Tables/orders", format="delta", mode="overwrite")
df_geolocation.write.save(f"{gold_lakehouse_abfsPath }/Tables/geolocation", format="delta", mode="overwrite")
df_sellers.write.save(f"{gold_lakehouse_abfsPath }/Tables/sellers", format="delta", mode="overwrite")
df_customers.write.save(f"{gold_lakehouse_abfsPath }/Tables/customers", format="delta", mode="overwrite")
df_order_pay.write.save(f"{gold_lakehouse_abfsPath }/Tables/order_pay", format="delta", mode="overwrite")
# marketing 
df_closed_deals.write.save(f"{gold_lakehouse_abfsPath }/Tables/closed_deals", format="delta", mode="overwrite")
df_qualified_leads.write.save(f"{gold_lakehouse_abfsPath }/Tables/qualified_leads", format="delta", mode="overwrite")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql import functions as F
from pyspark.sql import DataFrame

SUBSCRIPTION_KEY = "YOUR SUBSCRIPTION KEY"
LOCATION = "northeurope"
FROM_LANGUAGE = "pt"
TO_LANGUAGE = "en"

def translate_and_transform(df: DataFrame, text_col: str) -> DataFrame:

    translate = Translate(fromLanguage=FROM_LANGUAGE) \
        .setSubscriptionKey(SUBSCRIPTION_KEY) \
        .setLocation(LOCATION) \
        .setTextCol(text_col) \
        .setToLanguage(TO_LANGUAGE) \
        .setOutputCol("output_col")
    
    columns_to_select = df.columns

    df_transformed = translate.transform(df)
    
    df_transformed = df_transformed.withColumn(
        "translation", 
        F.flatten(F.col("output_col.translations"))
    ).withColumn(
        "translation", 
        F.col("translation.text")
    ).withColumn(
        "translation", 
        F.col("translation").getItem(0)
    )
    
    df_transformed = df_transformed.withColumn(text_col, F.col("translation"))
    
    df_transformed = df_transformed.select(columns_to_select)
    
    return df_transformed

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(df_reviews.select(F.length("review_comment_message"), F.length("review_comment_title")).summary())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

!pip install azure-ai-translation-text==1.0.0b1 -q

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# 900 
# 
# 45000
# 
# En batch med tekst kan maks bestå av 1000 strings (setninger) eller at det totalt sett er maks 50 000 characters.
# 
# 
# SynapseML azure ai translator, det fungerte, men den hang, det fungerte for small to medium data, funker nok veldig bra for 
# 
# Å batche det og å bruke azure ai translation fungerte best i vår tilfelle
# 
# 
# Erlend har også laget en rate limiter pga det er dette på apiet fra deres side.


# CELL ********************

from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
from azure.ai.translation.text.models import InputTextItem
from azure.core.exceptions import HttpResponseError
import time


credential = TranslatorCredential("YOUR SUBSCRIPTION KEY", LOCATION)
text_translator = TextTranslationClient(credential=credential)


def batch_inputs(rows: list):
    """Batch messages to optimize API"""
    # Initialize variables
    sublists = []
    sublist = []
    sublist_char_count = 0

    # Iterate over each tuple in the list
    for tup in rows:
        # Unpack the tuple into id and string
        idx, string = tup
        # If adding the current string would exceed the limits
        if (sublist_char_count + len(string)) > 45_000 or len(sublist) == 900:
            # Add the current sublist to the list of sublists
            sublists.append(sublist)
            # Start a new sublist and character count
            sublist = [(idx, InputTextItem(text=string))]
            sublist_char_count = len(string)
        else:
            # Add the current tuple to the current sublist and update the character count
            sublist.append((idx, InputTextItem(text=string)))
            sublist_char_count += len(string)

    # Add the last sublist if it's not empty
    if sublist:
        sublists.append(sublist)

    return sublists

def batch_translate(translator, text_elements: list, **translate_kwargs)->list:
    """Contains an id and the text column"""
    idxs, texts = zip(*text_elements)
    
    try:
        source_language = FROM_LANGUAGE
        target_languages = [TO_LANGUAGE]
        kwargs = {"to": [TO_LANGUAGE], "from_parameter" :FROM_LANGUAGE, **translate_kwargs}
        response = text_translator.translate(content = texts, **kwargs)
        tranlated_texts = [txt_obj["translations"][0]["text"] for txt_obj in response if response]
        return list(zip(idxs, [txt["text"] for txt in texts], tranlated_texts))

        if translation:
            for translated_text in translation.translations:
                print(f"Text was translated to: '{translated_text.to}' and the result is: '{translated_text.text}'.")
    except HttpResponseError as exception:
        print(f"Error Code: {exception.error.code}")
        print(f"Message: {exception.error.message}")

def add_translation(df, batch_translations:list, id_col:str, txt_col:str) -> DataFrame:
    flattened_translations = [record for batch in batch_translations for record in batch]
    translated_df = spark.createDataFrame(flattened_translations, [id_col, txt_col, f"{txt_col}_{TO_LANGUAGE}"])
    return df.drop(txt_col).join(translated_df.alias("translated"),
                                on=id_col,
                                how="left"
                            )


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
from azure.ai.translation.text.models import InputTextItem
from azure.core.exceptions import HttpResponseError

text_translator = TextTranslationClient(credential=credential)

def batch_translate_df(df, id_col:str, txt_col: str, translator, **translate_kwargs) -> tuple:
    txt_list = df.dropna(subset=[txt_col]).select(id_col, txt_col).collect()
    txt_batches = batch_inputs(txt_list)

    translated = []
    for idx, batch in enumerate(txt_batches, start=1):
        print("\tTranslated batch", idx, "out of", len(txt_batches))
        translated_batch = batch_translate(translator, batch, **translate_kwargs)
        while translated_batch is None:
            time.sleep(30)
            print("\tRetry batch", idx)
            translated_batch = batch_translate(translator, batch, **translate_kwargs)

        translated.append(translated_batch)
    return txt_batches, translated



title_batches, title_translations = batch_translate_df(df_reviews, "review_id", "review_comment_title", text_translator)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print("Translating review titles")
title_batches, title_translations = batch_translate_df(df_reviews, "review_id", "review_comment_title", text_translator)
title_translated_df = add_translation(df_reviews, title_translations, "review_id", "review_comment_title")
title_translated_df.write.option("overwriteSchema", "true").save(f"{gold_lakehouse_abfsPath }/Tables/reviews", format="delta", mode="overwrite")

print("Translating review messages")
message_batches, message_translations = batch_translate_df(title_translated_df, "review_id", "review_comment_message", text_translator)
title_message_translated_df = add_translation(translated_df, translations, "review_id", "review_comment_message")
title_message_translated_df.write.option("overwriteSchema", "true").save(f"{gold_lakehouse_abfsPath }/Tables/reviews", format="delta", mode="overwrite")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# # Translate review comment message
# df_reviews_translated = translate_and_transform(
#     df=df_reviews,
#     text_col="review_comment_message"
# )

# # Translate review comment title
# df_reviews_translated = translate_and_transform(
#     df=df_reviews_translated,
#     text_col="review_comment_title"
# )
# df_reviews_translated.cache()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Translate product category names
# Add missing categories
missing_categories = spark.createDataFrame([("pc_gamer", "gaming_pc"), ("portateis_cozinha_e_preparadores_de_alimentos", "portable_kitchen_and_food_preperators")], ["product_category_name", "product_category_name_english"])
df_category_extended = df_category.union(missing_categories)

# Include english translation
df_products_category = df_products.join(df_category_extended.alias("category"), on="product_category_name", how="left")
df_products_category

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_products_category.write.save(f"{gold_lakehouse_abfsPath }/Tables/products", format="delta", mode="overwrite")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
