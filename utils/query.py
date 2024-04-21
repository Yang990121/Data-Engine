from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud import storage
import streamlit as st
import io
import pickle

CREDS = 'is3107-418011-f63573e5e1f3.json'
credentials = service_account.Credentials.from_service_account_file(CREDS)
client = bigquery.Client(credentials=credentials)
storage_client = storage.Client(credentials=credentials)


@st.cache_data(ttl=3600, show_spinner=True)
def query_table_from_bq_filtered(town_types):
    # Ensure town_types is a list of town names
    # Convert all town names to lowercase before passing to the query
    lower_town_types = [town.lower() for town in town_types]

    query = """
            SELECT 
                LOWER(p.flat_model) as flat_model, 
                LOWER(a.town) as town, 
                AVG(d.resale_price) as resale_price,
                LAST_DAY(PARSE_DATE('%Y-%m', t.month)) as date
            FROM `is3107-418011.is3107.resale_data` d
            LEFT JOIN `is3107-418011.is3107.Property_new` p ON d.property_id = p.property_id
            LEFT JOIN `is3107-418011.is3107.Address_new` a ON d.address_id = a.address_id
            LEFT JOIN `is3107-418011.is3107.Transaction_new` t ON d.transaction_id = t.transaction_id
            WHERE LOWER(a.town) IN UNNEST(@towns)
            GROUP BY flat_model, town, date
        """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("towns", "STRING", lower_town_types)
        ]
    )

    # Execute the query with the specified parameter
    return client.query(query, job_config=job_config).to_dataframe()


@st.cache_data(ttl=3600, show_spinner=True)
def query_table_from_bq_old(town_type):
    # Assuming `flat_model_type` is a string containing the model type to filter for
    # Using parameterized queries for safety and efficiency
    query = """
            SELECT 
                LOWER(p.flat_model) as flat_model, 
                LOWER(a.town) as town, 
                AVG(d.resale_price) as resale_price,
                LAST_DAY(PARSE_DATE('%Y-%m', t.month)) as date
            FROM `is3107-418011.is3107.resale_data` d
            LEFT JOIN `is3107-418011.is3107.Property_new` p ON d.property_id = p.property_id
            LEFT JOIN `is3107-418011.is3107.Address_new` a ON d.address_id = a.address_id
            LEFT JOIN `is3107-418011.is3107.Transaction_new` t ON d.transaction_id = t.transaction_id
            WHERE LOWER(a.town) = @town
            GROUP BY flat_model, town, date
        """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("town", "STRING", town_type.lower())
        ]
    )

    # Execute the query with the specified parameter
    return client.query(query, job_config=job_config).to_dataframe()


@st.cache_resource(ttl=3600, show_spinner=True)
def load_model_from_gcs(model):
    # Get the bucket and blob
    bucket_name = 'is3107_bucket'
    pickle_file_name = model
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(pickle_file_name)

    # Download the blob to a bytes buffer
    byte_stream = io.BytesIO()
    blob.download_to_file(byte_stream)
    byte_stream.seek(0)

    # Load and return the model
    model = pickle.load(byte_stream)
    return model
