import json
import requests
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import re
from datetime import datetime
from airflow.decorators import dag, task
import os
from airflow.providers.postgres.hooks.postgres import PostgresHook
from google.cloud import bigquery
from google.oauth2 import service_account
import time


# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'start_date': datetime(2024, 1, 1),
}


@dag(
    dag_id='data_refresh_dag',
    default_args=default_args,
    description='A DAG for refresh machine learning model for IS3107 project',
    schedule=None,
    catchup=False,
    tags=['is3107project'],
)

def model_refresh_etl():

    @task() 
    def feature_engineering(download_path: str):
        
        csv_file_path = os.path.join(download_path, 'processed_data/filtered_df3.csv')
        
        df1 = pd.read_csv(csv_file_path)
        threshold = 0.05*len(df1)


        df1['month'] = pd.to_datetime(df1['month'])
        df1 = df1[df1['month'] > '2015-01-01']
        
        # use month and lease_commence_date to calculate the age of the flat
        df1['month'] = pd.to_datetime(df1['month'])
        df1['lease_commence_date'] = pd.to_datetime(df1['lease_commence_date'])

        df1.loc[:,'age_of_flat'] = df1['month'].dt.year - df1['lease_commence_date'].dt.year
        df1 = df1.sort_values('month')
        
        columns_of_interest = ['avg_storey_range', 'floor_area_sqm', 'normalized_resale_price',
       'total_dwelling_units', 'vacancy', 'commercial', 
       'mrt_interchange', 'flat_model', 'town', 'age_of_flat']
        
        df1 = df1[columns_of_interest]

        df1['flat_model'] = df1['flat_model'].str.lower()
        df1['town'] = df1['town'].str.lower()
        
        # find the flat_models that have less than 10,000
        other_models = df1['flat_model'].value_counts()[df1['flat_model'].value_counts() < threshold].index
        other_models = list(other_models)

        # replace the flat_models that have less than 10,000 with 'other'
        df1['flat_model'] = df1['flat_model'].replace(other_models, 'other')
        
        
        # find the flat_types that have less than 10,000
        other_models = df1['town'].value_counts()[df1['town'].value_counts() < threshold].index
        other_models = list(other_models)

        # replace the flat_models that have less than 10,000 with 'other'
        df1['town'] = df1['town'].replace(other_models, 'other')
        
        
        columns_to_be_scaled = ['avg_storey_range', 
       'floor_area_sqm',
       'normalized_resale_price', 
       'total_dwelling_units', 'vacancy']
        
        
        scaling_dic = {}

        for col in columns_to_be_scaled:
            col_min = min(df1[col])
            col_max = max(df1[col])
            df1[col] = df1[col].apply(lambda x: (x-col_min)/(col_max-col_min))
            scaling_dic[col] = (col_min, col_max)

        json_file_path = os.path.join(download_path, 'processed_data/scaling_info.json')

        # Store scaling_dic in JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(scaling_dic, json_file)
            
        return download_path
    
    download_path = feature_engineering("/Users/renzhou/Downloads/Y3S2/IS3107/Data-Engine/ETL/01_Data_Collection/01_dataset")

data_collection_dag = model_refresh_etl()
        