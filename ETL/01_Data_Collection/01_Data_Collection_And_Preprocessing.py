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
    dag_id='data_collection_dag',
    default_args=default_args,
    description='A DAG for extracting data for IS3107 project',
    schedule=None,
    catchup=False,
    tags=['is3107project'],
)
def data_collection_etl():
    @task
    def extract_external_data():
        urls = ['https://data.gov.sg/api/action/datastore_search?resource_id=d_ebc5ab87086db484f88045b47411ebc5',
                'https://data.gov.sg/api/action/datastore_search?resource_id=d_43f493c6c50d54243cc1eab0df142d6a',
                'https://data.gov.sg/api/action/datastore_search?resource_id=d_2d5ff9ea31397b66239f245f57751537',
                'https://data.gov.sg/api/action/datastore_search?resource_id=d_ea9ed51da2787afaf8e51f827c304208',
                'https://data.gov.sg/api/action/datastore_search?resource_id=d_8b84c4ee58e3cfc0ece0d773c8ca6abc',
                ]

        csv_names = [
            'resale_flat_prices_1990-1999_new.csv',
            'resale_flat_prices_2000-2012_new.csv',
            'resale_flat_prices_2012-2014_new.csv',
            'resale_flat_prices_2015-2016_new.csv',
            'resale_flat_prices_2017-2024_new.csv',
        ]
        for url, csv_name in zip(urls, csv_names):
            offset = 0
            limit = 10000  # Adjust the limit based on your requirements

            # Lists to store data
            town = []
            flat_type = []
            flat_model = []
            floor_area_sqm = []
            street_name = []
            resale_price = []
            month = []
            remaining_lease = []
            lease_commence_date = []
            storey_range = []
            _id = []
            block = []

            # Flag to control the loop
            has_next = True

            while has_next:
                # Construct the URL with offset and limit
                request_url = f'{url}&offset={offset}&limit={limit}'

                # Send a GET request to the API endpoint
                response = requests.get(request_url)
                # Check if the request was successful (status code 200)
                if response.status_code == 200:
                    # Extract the JSON data from the response
                    data = response.json()
                    # print(data)
                    # Check if there are records in the response
                    if data['result']['records']:
                        hdb_price_dict_records = data['result']['records']

                    # Iterate through the records and append data to lists
                        for record in hdb_price_dict_records:
                            town.append(record['town'])
                            flat_type.append(record['flat_type'])
                            flat_model.append(record['flat_model'])
                            floor_area_sqm.append(record['floor_area_sqm'])
                            street_name.append(record['street_name'])
                            resale_price.append(record['resale_price'])
                            month.append(record['month'])
                            lease_commence_date.append(
                                record['lease_commence_date'])
                            storey_range.append(record['storey_range'])
                            _id.append(record['_id'])
                            block.append(record['block'])
                            if 'remaining_lease' in record:
                                remaining_lease.append(
                                    record['remaining_lease'])

                        # Update the offset for the next iteration
                        offset += limit
                        print(offset)
                    else:
                        # No more records, exit the loop
                        has_next = False
                else:
                    # Print an error message if the request was not successful
                    print(
                        f"Error: Unable to fetch data. Status code: {response.status_code}")
                    has_next = False  # Exit the loop on error

            # Check if _id and remaining_lease lists are empty
            if _id and remaining_lease:
                # Include _id and remaining_lease in data_dict
                data_dict = {
                    '_id': _id,
                    'month': month,
                    'town': town,
                    'flat_type': flat_type,
                    'block': block,
                    'street_name': street_name,
                    'storey_range': storey_range,
                    'floor_area_sqm': floor_area_sqm,
                    'flat_model': flat_model,
                    'lease_commence_date': lease_commence_date,
                    'resale_price': resale_price,
                    'remaining_lease': remaining_lease,
                }
            else:
                # Exclude _id and remaining_lease from data_dict
                data_dict = {
                    'month': month,
                    'town': town,
                    'flat_type': flat_type,
                    'block': block,
                    'street_name': street_name,
                    'storey_range': storey_range,
                    'floor_area_sqm': floor_area_sqm,
                    'flat_model': flat_model,
                    'lease_commence_date': lease_commence_date,
                    'resale_price': resale_price,
                }
            # Create DataFrame
            df = pd.DataFrame(data_dict)

            # Define the file path for the CSV file
            download_path = os.path.join(os.getcwd(), '01_dataset')
            if not os.path.exists(download_path):
                os.makedirs(download_path)
            download_path_2 = os.path.join(download_path, 'new_api_dataset')
            if not os.path.exists(download_path_2):
                os.makedirs(download_path_2)

            csv_file_path = os.path.join(download_path_2, csv_name)  # Provide the desired file path

            # Export the DataFrame to a CSV file
            df.to_csv(csv_file_path, index=False)  # Set index=False to exclude the index from the CSV file

            print(f"DataFrame has been exported to {csv_file_path}")  
        download_path = '/Users/renzhou/Downloads/Y3S2/IS3107/Data-Engine/ETL/01_Data_Collection/01_dataset'

        return download_path

    @task
    def data_combination(download_path: str):
        # Initialize an empty list to store DataFrames
        dfs = []
        download_path_2 = os.path.join(download_path, 'new_api_dataset')

        # Loop through files in the download path
        for filename in os.listdir(download_path_2):
            if filename.endswith(".csv"):
                # Read each CSV file and append it to the list of DataFrames
                df = pd.read_csv(os.path.join(download_path_2, filename))
                dfs.append(df)

        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)

        # Check if '_id' column exists and drop it
        if '_id' in combined_df.columns:
            combined_df.drop(columns=['_id'], inplace=True)

        # Check if 'remaining_lease' column exists and drop it
        if 'remaining_lease' in combined_df.columns:
            combined_df.drop(columns=['remaining_lease'], inplace=True)
            
        # Define the file path for the CSV file
        csv_name = 'combined_dataframe.csv'
        csv_file_path = os.path.join(download_path, 'processed_data')
        if not os.path.exists(csv_file_path):
                os.makedirs(csv_file_path)
        csv_file_path_2 = os.path.join(csv_file_path, csv_name)

        # Export the DataFrame to a CSV file
        combined_df.to_csv(csv_file_path_2, index=False)

        return download_path
    
    @task
    def process_external_data(download_path: str):
        # Load the dataset
        csv_file_path = os.path.join(download_path, 'downloaded_data/resale-flat-prices-full-version.csv')
        df = pd.read_csv(csv_file_path)

        # Select columns 14 to 41 from the original DataFrame
        feature_dict = df.iloc[:, [14] + list(range(16, 41))]

        # Remove duplicate rows from the DataFrame
        feature_dict_no_duplicates = feature_dict.drop_duplicates()

        # Define the file path for the CSV file
        csv_output_file_path = os.path.join(download_path,'processed_data/feature_dict.csv')

        # Export the DataFrame to a CSV file
        feature_dict_no_duplicates.to_csv(csv_output_file_path, index=False)

        return download_path
        
    @task
    def feature_engineering(download_path: str):
        dict_file_path = os.path.join(download_path, 'processed_data/feature_dict.csv')
        df_file_path = os.path.join(download_path, 'processed_data/combined_dataframe.csv')
        
        # Read the CSV files into DataFrames
        feature_dict = pd.read_csv(dict_file_path)
        df = pd.read_csv(df_file_path)

        # Create a new column by concatenating 'block' and 'street_name'
        df['address'] = df['block'] + ', ' + df['street_name']

        # Perform an inner join between df and feature_dict on the 'address' column
        merged_df = pd.merge(df, feature_dict, on='address', how='inner')
        
        # Split 'month' column into 'year' and 'quarter'
        merged_df[['year', 'quarter']] = merged_df['month'].str.split('-', expand=True)

        # Define a function to map month to quarter
        def map_to_quarter(month):
            month_int = int(month)
            if 1 <= month_int <= 3:
                return '1'
            elif 4 <= month_int <= 6:
                return '2'
            elif 7 <= month_int <= 9:
                return '3'
            else:
                return '4'

        # Apply the function to map month to quarter
        merged_df['quarter'] = merged_df['quarter'].apply(map_to_quarter)
        
        merged_df['year']=merged_df['year'].astype(int)
        
        filtered_df = merged_df[(merged_df['year'] >= 1998) & ((merged_df['year'] == 2023) & (merged_df['quarter'] != 4) | (merged_df['year'] < 2023))]
        
        # Print out the unique values in the 'storey_range' column
        unique_storey_ranges = filtered_df['storey_range'].unique()
        
        # Clean the storey_range data and calculate average range
        avg_storey_ranges = {}

        for range_str in unique_storey_ranges:
            start, end = map(int, range_str.split(' TO '))
            avg_storey = (start + end) / 2
            avg_storey_ranges[range_str] = avg_storey
    
    
        # Add a new column 'avg_storey_range' using the dictionary mapping
        filtered_df['avg_storey_range'] = filtered_df['storey_range'].map(avg_storey_ranges)
        
        
        # Combine 'flat_type' and 'flat_model' columns into a new column 'flat_type_model'
        filtered_df['flat_type_model'] = filtered_df['flat_type'] + ' ' + filtered_df['flat_model']
        
        
        categories = {
            'neighbourhood': (-float('inf'), 210),
            'good': (200, 245),
            'elite': (245, float('inf'))
        }

        # Define a function to categorize cutoff_point values
        def categorize_cutoff(cutoff):
            for category, (lower, upper) in categories.items():
                if lower <= cutoff < upper:
                    return category

        # Apply the function to create the 'school_type' column
        filtered_df['school_type'] = filtered_df['cutoff_point'].apply(categorize_cutoff)
        
        
        # Define the cutoff point ranges and corresponding categories
        proximity_categories = {
            'within 3 minutes': (-float('inf'), 80),
            '3-5 minutes': (80, 240),
            '5-10 minutes': (240, 500),
            '10-15 minutes': (500, 1000),
            'more than 15 minutes': (1000, float('inf'))
        }

        # Define a function to categorize bus_stop_nearest_distance values
        def categorize_proximity(cutoff):
            for category, (lower, upper) in proximity_categories.items():
                if lower <= cutoff < upper:
                    return category

        # Apply the function to create the 'bus_stop_proximity' column
        filtered_df['bus_stop_proximity'] = filtered_df['bus_stop_nearest_distance'].apply(categorize_proximity)
        filtered_df['mrt_proximity'] = filtered_df['mrt_nearest_distance'].apply(categorize_proximity)
        
        # Apply the function to create the 'sch_proximity' column
        filtered_df['pri_sch_proximity'] = filtered_df['pri_sch_nearest_distance'].apply(categorize_proximity)
        filtered_df['sec_sch_proximity'] = filtered_df['sec_sch_nearest_dist'].apply(categorize_proximity)
        
        # convert data type to string
        for column in filtered_df.columns:
            if filtered_df[column].dtype not in ['float64', 'int64']:
                filtered_df[column] = filtered_df[column].astype(str)
                
        # Define the file path for the CSV file
        csv_output_file_path = os.path.join(download_path,'processed_data/filtered_df1.csv')

        # Export the DataFrame to a CSV file
        filtered_df.to_csv(csv_output_file_path, index=False)
        
        return download_path

    @task
    def get_location(download_path: str):
        dict_file_path = os.path.join(download_path, 'downloaded_data/location.csv')
        df_file_path = os.path.join(download_path, 'processed_data/filtered_df1.csv')
        
        # Load resale index data
        location_df = pd.read_csv(dict_file_path, index_col=0)
        filtered_df = pd.read_csv(df_file_path)

        # Merge DataFrames
        merged_df = pd.merge(filtered_df, location_df, on='postal', how='left')
        
        # Define the file path for the CSV file
        csv_output_file_path = os.path.join(download_path,'processed_data/filtered_df2.csv')

        # Export the DataFrame to a CSV file
        merged_df.to_csv(csv_output_file_path, index=False)
        
        return download_path



    @task
    def normalize_price(download_path: str):
        dict_file_path = os.path.join(download_path, 'downloaded_data/QSGR628BIS.csv')
        df_file_path = os.path.join(download_path, 'processed_data/filtered_df2.csv')
        
        # Load resale index data
        resale_index = pd.read_csv(dict_file_path)
        resale_index['DATE'] = pd.to_datetime(resale_index['DATE'])

        # Extract year and month into separate columns
        resale_index['year'] = resale_index['DATE'].dt.year
        resale_index['month'] = resale_index['DATE'].dt.month

        # Define a function to map month values to quarter values
        def month_to_quarter(month):
            if month == 1:
                return 1
            elif month == 4:
                return 2
            elif month == 7:
                return 3
            else:
                return 4
        
        # Apply the function to create the 'quarter' column
        resale_index['quarter'] = resale_index['month'].apply(month_to_quarter)
        resale_index['resale_index'] = resale_index['QSGR628BIS'].astype(float)
        
        # Calculate current resale index for 2023, quarter 4
        resale_index_current = resale_index[(resale_index['year'] == 2023) & (resale_index['quarter'] == 4)]['resale_index'].iloc[0]
        
        # Calculate inflation
        resale_index['inflation'] = ((resale_index_current - resale_index['resale_index']) / resale_index['resale_index']) + 1
        
        # Load filtered DataFrame
        filtered_df = pd.read_csv(df_file_path)
        filtered_df['quarter'] = filtered_df['quarter'].astype(int)
        
        # Merge with resale index data on year and quarter
        filtered_df = pd.merge(filtered_df, resale_index[['year', 'quarter', 'inflation']], on=['year', 'quarter'], how='left')
        
        # Calculate normalized resale price
        filtered_df['normalized_resale_price'] = filtered_df['inflation'] * filtered_df['resale_price']
        
        filtered_df['index'] = filtered_df.reset_index().index
        
        # Define the file path for the CSV file
        csv_output_file_path = os.path.join(download_path,'processed_data/filtered_df3.csv')

        # Export the DataFrame to a CSV file
        filtered_df.to_csv(csv_output_file_path, index=False)
        
        return download_path
    
    @task
    def create_tables():
        create_property_sql = """
        DROP TABLE IF EXISTS Property;

        CREATE TABLE Property (
            index INTEGER PRIMARY KEY,
            postal INTEGER,
            flat_type VARCHAR(50),
            flat_model VARCHAR(50),
            flat_type_model VARCHAR(50),
            floor_area_sqm FLOAT,
            lease_commence_date INTEGER,
            storey_range VARCHAR(20),
            avg_storey_range FLOAT,
            total_dwelling_units INTEGER
        );
        """
        
        create_address_sql = """
        DROP TABLE IF EXISTS Address;

        CREATE TABLE Address (
            postal INTEGER PRIMARY KEY,
            address VARCHAR(255) ,
            town VARCHAR(255),
            block VARCHAR(10),
            street_name VARCHAR(255),
            planning_area VARCHAR(255),
            latitude FLOAT,
            longitude FLOAT
        );
        """
        
        create_surroundings_sql = """
        DROP TABLE IF EXISTS Surroundings;

        CREATE TABLE Surroundings (
            postal INTEGER PRIMARY KEY,
            commercial INTEGER,
            market_hawker INTEGER,
            multistorey_carpark INTEGER,
            precinct_pavilion INTEGER,
            total_dwelling_units INTEGER,
            mall_nearest_distance FLOAT,
            hawker_nearest_distance FLOAT,
            hawker_food_stalls INTEGER,
            hawker_market_stalls INTEGER,
            mrt_nearest_distance FLOAT,
            mrt_name VARCHAR(255),
            bus_interchange INTEGER,
            mrt_interchange INTEGER,
            bus_stop_nearest_distance FLOAT,
            bus_stop_name VARCHAR(255),
            pri_sch_nearest_distance FLOAT,
            pri_sch_name VARCHAR(255),
            vacancy INTEGER,
            pri_sch_affiliation INTEGER,
            sec_sch_nearest_dist FLOAT,
            sec_sch_name VARCHAR(255),
            cutoff_point INTEGER,
            affiliation INTEGER,
            school_type VARCHAR(50),
            bus_stop_proximity VARCHAR(50),
            mrt_proximity VARCHAR(50),
            pri_sch_proximity VARCHAR(50),
            sec_sch_proximity VARCHAR(50)
        );
        """
        
        create_transaction_sql = """
        DROP TABLE IF EXISTS Transaction;

        CREATE TABLE Transaction (
            index INTEGER PRIMARY KEY,
            month VARCHAR(50),
            resale_price FLOAT,
            year INTEGER,
            quarter INTEGER,             
            inflation FLOAT,
            normalized_resale_price FLOAT
        );
        """
        
        # Use the Airflow PostgresHook to execute the SQL
        hook = PostgresHook(postgres_conn_id='is3107_project')
        hook.run(create_property_sql, autocommit=True)
        hook.run(create_address_sql, autocommit=True)
        hook.run(create_surroundings_sql, autocommit=True)
        hook.run(create_transaction_sql, autocommit=True)

    @task
    def create_main_table():
        create_table_sql = """
        DROP TABLE IF EXISTS resale_flat_prices;
        CREATE TABLE IF NOT EXISTS resale_flat_prices (
            index INTEGER PRIMARY KEY,
            month VARCHAR(50),
            town VARCHAR(255),
            flat_type VARCHAR(50),
            block VARCHAR(10),
            street_name VARCHAR(255),
            storey_range VARCHAR(20),
            floor_area_sqm FLOAT,
            flat_model VARCHAR(50),
            lease_commence_date INTEGER,
            resale_price FLOAT,
            address VARCHAR(255),
            commercial INTEGER,
            market_hawker INTEGER,
            multistorey_carpark INTEGER,
            precinct_pavilion INTEGER,
            total_dwelling_units INTEGER,
            postal INTEGER,
            planning_area VARCHAR(255),
            mall_nearest_distance FLOAT,
            hawker_nearest_distance FLOAT,
            hawker_food_stalls INTEGER,
            hawker_market_stalls INTEGER,
            mrt_nearest_distance FLOAT,
            mrt_name VARCHAR(255),
            bus_interchange INTEGER,
            mrt_interchange INTEGER,
            bus_stop_nearest_distance FLOAT,
            bus_stop_name VARCHAR(255),
            pri_sch_nearest_distance FLOAT,
            pri_sch_name VARCHAR(255),
            vacancy INTEGER,
            pri_sch_affiliation INTEGER,
            sec_sch_nearest_dist FLOAT,
            sec_sch_name VARCHAR(255),
            cutoff_point INTEGER,
            affiliation INTEGER,
            year INTEGER,
            quarter INTEGER,
            avg_storey_range FLOAT,
            flat_type_model VARCHAR(50),
            school_type VARCHAR(50),
            bus_stop_proximity VARCHAR(50),
            mrt_proximity VARCHAR(50),
            pri_sch_proximity VARCHAR(50),
            sec_sch_proximity VARCHAR(50),
            inflation FLOAT,
            latitude FLOAT,
            longitude FLOAT,
            normalized_resale_price FLOAT
        );
        """
        # Use the Airflow PostgresHook to execute the SQL
        hook = PostgresHook(postgres_conn_id='is3107_project')
        hook.run(create_table_sql, autocommit=True)
        
    @task
    def load(download_path: str):
        # Read the CSV file into a DataFrame
        df_file_path = os.path.join(download_path, 'processed_data/filtered_df3.csv')
        
        # Read the CSV files into DataFrames
        df = pd.read_csv(df_file_path)

        # Convert DataFrame to a list of dictionaries (each dictionary represents a row)
        data_to_insert = df.to_dict(orient='records')
        
        # Define the SQL query for inserting data into the table
        load_sql_property = """
        INSERT INTO Property (
            index,
            postal,
            flat_type,
            flat_model,
            flat_type_model,
            floor_area_sqm,
            lease_commence_date,
            storey_range,
            avg_storey_range,
            total_dwelling_units
        ) VALUES (
            %(index)s,
            %(postal)s, 
            %(flat_type)s,
            %(flat_model)s, 
            %(flat_type_model)s, 
            %(floor_area_sqm)s, 
            %(lease_commence_date)s,
            %(storey_range)s, 
            %(avg_storey_range)s, 
            %(total_dwelling_units)s
        ) ON CONFLICT (index) DO NOTHING;
        """
        
        
        load_sql_address = """
        INSERT INTO Address (
            postal,
            address,
            town,
            block,
            street_name,
            planning_area,
            latitude,
            longitude
        ) VALUES (
            %(postal)s, 
            %(address)s, 
            %(town)s, 
            %(block)s, 
            %(street_name)s, 
            %(planning_area)s, 
            %(latitude)s, 
            %(longitude)s
        ) ON CONFLICT (postal) DO NOTHING;
        """
        
        
        load_sql_surroundings = """
        INSERT INTO Surroundings (
            postal,
            commercial,
            market_hawker,
            multistorey_carpark,
            precinct_pavilion,
            total_dwelling_units,
            mall_nearest_distance,
            hawker_nearest_distance,
            hawker_food_stalls,
            hawker_market_stalls,
            mrt_nearest_distance,
            mrt_name,
            bus_interchange,
            mrt_interchange,
            bus_stop_nearest_distance,
            bus_stop_name,
            pri_sch_nearest_distance,
            pri_sch_name,
            vacancy,
            pri_sch_affiliation,
            sec_sch_nearest_dist,
            sec_sch_name,
            cutoff_point,
            affiliation,
            school_type,
            bus_stop_proximity,
            mrt_proximity,
            pri_sch_proximity,
            sec_sch_proximity
        ) VALUES (
            %(postal)s, 
            %(commercial)s, 
            %(market_hawker)s,
            %(multistorey_carpark)s,
            %(precinct_pavilion)s, 
            %(total_dwelling_units)s,
            %(mall_nearest_distance)s, 
            %(hawker_nearest_distance)s, 
            %(hawker_food_stalls)s,
            %(hawker_market_stalls)s,
            %(mrt_nearest_distance)s,
            %(mrt_name)s, 
            %(bus_interchange)s,
            %(mrt_interchange)s, 
            %(bus_stop_nearest_distance)s,
            %(bus_stop_name)s, 
            %(pri_sch_nearest_distance)s, 
            %(pri_sch_name)s, 
            %(vacancy)s, 
            %(pri_sch_affiliation)s,
            %(sec_sch_nearest_dist)s, 
            %(sec_sch_name)s, 
            %(cutoff_point)s, 
            %(affiliation)s,
            %(school_type)s, 
            %(bus_stop_proximity)s, 
            %(mrt_proximity)s,
            %(pri_sch_proximity)s, 
            %(sec_sch_proximity)s
        ) ON CONFLICT (postal) DO NOTHING;
        """
        
        
        load_sql_transaction = """
        INSERT INTO Transaction (
            index,
            month,
            resale_price,
            year,
            quarter,             
            inflation,
            normalized_resale_price
        ) VALUES (
            %(index)s, 
            %(month)s, 
            %(resale_price)s,
            %(year)s, 
            %(quarter)s,
            %(inflation)s,
            %(normalized_resale_price)s
        ) ON CONFLICT (index) DO NOTHING;
        """
        
        load_sql = """
        INSERT INTO resale_flat_prices (
            index, month, town, flat_type, block, street_name, storey_range, floor_area_sqm,
            flat_model, lease_commence_date, resale_price, address, commercial, market_hawker,
            multistorey_carpark, precinct_pavilion, total_dwelling_units, postal, planning_area,
            mall_nearest_distance, hawker_nearest_distance, hawker_food_stalls, hawker_market_stalls,
            mrt_nearest_distance, mrt_name, bus_interchange, mrt_interchange, bus_stop_nearest_distance,
            bus_stop_name, pri_sch_nearest_distance, pri_sch_name, vacancy, pri_sch_affiliation,
            sec_sch_nearest_dist, sec_sch_name, cutoff_point, affiliation, year, quarter, avg_storey_range,
            flat_type_model, school_type, bus_stop_proximity, mrt_proximity, pri_sch_proximity,
            sec_sch_proximity, inflation, latitude, longitude, normalized_resale_price
        ) VALUES (
            %(index)s, %(month)s, %(town)s, %(flat_type)s, %(block)s, %(street_name)s, %(storey_range)s, %(floor_area_sqm)s,
            %(flat_model)s, %(lease_commence_date)s, %(resale_price)s, %(address)s, %(commercial)s, %(market_hawker)s,
            %(multistorey_carpark)s, %(precinct_pavilion)s, %(total_dwelling_units)s, %(postal)s, %(planning_area)s,
            %(mall_nearest_distance)s, %(hawker_nearest_distance)s, %(hawker_food_stalls)s, %(hawker_market_stalls)s,
            %(mrt_nearest_distance)s, %(mrt_name)s, %(bus_interchange)s, %(mrt_interchange)s, %(bus_stop_nearest_distance)s,
            %(bus_stop_name)s, %(pri_sch_nearest_distance)s, %(pri_sch_name)s, %(vacancy)s, %(pri_sch_affiliation)s,
            %(sec_sch_nearest_dist)s, %(sec_sch_name)s, %(cutoff_point)s, %(affiliation)s, %(year)s, %(quarter)s,
            %(avg_storey_range)s, %(flat_type_model)s, %(school_type)s, %(bus_stop_proximity)s, %(mrt_proximity)s,
            %(pri_sch_proximity)s, %(sec_sch_proximity)s, %(inflation)s, %(latitude)s, %(longitude)s, %(normalized_resale_price)s
        ) ON CONFLICT (index) DO NOTHING;
        """
        
        # Use the Airflow PostgresHook to execute the SQL
        hook = PostgresHook(postgres_conn_id='is3107_project')
        for row in data_to_insert:
                hook.run(load_sql_property, parameters=row, autocommit=True)
                hook.run(load_sql_address, parameters=row, autocommit=True)
                hook.run(load_sql_surroundings, parameters=row, autocommit=True)
                hook.run(load_sql_transaction, parameters=row, autocommit=True)
                hook.run(load_sql, parameters=row, autocommit=True)
                
    @task 
    def load_on_cloud(download_path: str):
        # Read the CSV file into a DataFrame
        df_file_path = os.path.join(download_path, 'processed_data/filtered_df3.csv')
        
        CREDS = os.path.join(download_path, 'is3107-418011-f63573e5e1f3.json')
        credentials = service_account.Credentials.from_service_account_file(CREDS)
        client = bigquery.Client(credentials=credentials)
        job_config = bigquery.QueryJobConfig()
        
        # Read the CSV files into DataFrames
        df = pd.read_csv(df_file_path)
        
        property_id = ['flat_type', 'flat_model', 'flat_type_model',
                  'floor_area_sqm', 'lease_commence_date', 'storey_range',
                  'avg_storey_range', 'total_dwelling_units']
        
        address_id = ['postal','address', 'town', 'block', 'street_name', 
                                'planning_area', 'latitude', 'longitude']
        
        surroundings_id = ['commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion',
                        'mall_nearest_distance', 'hawker_nearest_distance',
                                    'hawker_food_stalls', 'hawker_market_stalls', 'mrt_nearest_distance', 'mrt_name', 'bus_interchange', 'mrt_interchange', 'bus_stop_nearest_distance', 'bus_stop_name',
                                    'pri_sch_nearest_distance', 'pri_sch_name', 'vacancy', 'pri_sch_affiliation',
                                    'sec_sch_nearest_dist', 'sec_sch_name', 'cutoff_point', 'affiliation', 'school_type',
                                    'bus_stop_proximity', 'mrt_proximity', 'pri_sch_proximity', 'sec_sch_proximity']
        transaction_id = ['month', 'year', 'quarter', 'inflation', 'normalized_resale_price']


        property_df = df[property_id].copy().drop_duplicates()
        property_df["property_id"] = [i for i in range(1 , len(property_df) + 1)]

        address_df = df[address_id].copy().drop_duplicates()
        address_df["address_id"] = [i for i in range(1 , len(address_df) + 1)]

        surroundings_df = df[surroundings_id].copy().drop_duplicates()
        surroundings_df["surroundings_id"] = [i for i in range(1 , len(surroundings_df) + 1)]

        transaction_df = df[transaction_id].copy().drop_duplicates()
        transaction_df["transaction_id"] = [i for i in range(1 , len(transaction_df) + 1)]

        resale_data = df.copy()
        resale_data["property_id"] = resale_data[property_id].merge(property_df, how='left')['property_id']
        resale_data["address_id"] = resale_data[address_id].merge(address_df, how='left')['address_id']
        resale_data["surroundings_id"] = resale_data[surroundings_id].merge(surroundings_df, how='left')['surroundings_id']
        resale_data["transaction_id"] = resale_data[transaction_id].merge(transaction_df, how='left')['transaction_id']

        # Drop the original columns that have been replaced
        resale_data.drop(property_id + address_id + surroundings_id + transaction_id, axis=1, inplace=True)
 
        project_id = "is3107-418011"

        # Dataset ID and table name
        dataset_id = "is3107"  # You have nested dataset, so specify only the innermost dataset

        table_id = "resale_price"   # Table ID within the specified dataset
        df.to_gbq(destination_table=f"{project_id}.{dataset_id}.{table_id}", project_id=project_id, if_exists="replace")
        
        # Upload DataFrame to BigQuery
        table_id = "resale_data"   # Table ID within the specified dataset
        resale_data.to_gbq(destination_table=f"{project_id}.{dataset_id}.{table_id}", project_id=project_id, if_exists="replace")
        
        # Dataset ID and table name
        table_id = "Property_new"   # Table ID within the specified dataset

        # Upload DataFrame to BigQuery
        property_df.to_gbq(destination_table=f"{project_id}.{dataset_id}.{table_id}", project_id=project_id, if_exists="replace")
        
        # Dataset ID and table name
        table_id = "Address_new"   # Table ID within the specified dataset

        # Upload DataFrame to BigQuery
        address_df.to_gbq(destination_table=f"{project_id}.{dataset_id}.{table_id}", project_id=project_id, if_exists="replace")
        
        # Dataset ID and table name
        table_id = "Surroundings_new"   # Table ID within the specified dataset

        # Upload DataFrame to BigQuery
        surroundings_df.to_gbq(destination_table=f"{project_id}.{dataset_id}.{table_id}", project_id=project_id, if_exists="replace")
        
        # Dataset ID and table name
        table_id = "Transaction_new"   # Table ID within the specified dataset

        # Upload DataFrame to BigQuery
        transaction_df.to_gbq(destination_table=f"{project_id}.{dataset_id}.{table_id}", project_id=project_id, if_exists="replace")

        return download_path       
                
    download_path = extract_external_data()
    download_path_2 = data_combination(download_path)
    download_path_3 = process_external_data(download_path_2)
    download_path_4 = feature_engineering(download_path_3)
    download_path_5 = get_location(download_path_4)
    download_path_6 = normalize_price(download_path_5)
    
    # create_tables()
    # create_main_table()
    # load(download_path_6)
    load_on_cloud(download_path_6)
    
    
                
data_collection_dag = data_collection_etl()