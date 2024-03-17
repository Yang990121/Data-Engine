import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime
from airflow.decorators import dag, task
import os
from airflow.providers.postgres.hooks.postgres import PostgresHook


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
            csv_file_path = os.path.join(download_path, csv_name)  # Provide the desired file path

            # Export the DataFrame to a CSV file
            df.to_csv(csv_file_path, index=False)  # Set index=False to exclude the index from the CSV file

            print(f"DataFrame has been exported to {csv_file_path}")   
extract_external_data()