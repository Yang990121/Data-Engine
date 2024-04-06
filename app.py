import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
from st_files_connection import FilesConnection
from postgres_funcs.postgres_loader import DatabaseManager
from google.cloud import bigquery
from google.oauth2 import service_account
import joblib
from utils.model_func import prediction_price

st.set_page_config(layout="wide")


def query_table_from_bq(table_name):
    CREDS = 'is3107-418011-f63573e5e1f3.json'
    credentials = service_account.Credentials.from_service_account_file(CREDS)
    client = bigquery.Client(credentials=credentials)
    job_config = bigquery.QueryJobConfig()

    # Set the destination table
    table = client.dataset(dataset_id='is3107-418011.is3107').table(table_id=f"is3107-418011.is3107.{table_name}")
    job_config.destination = table
    query = f"SELECT * FROM `is3107-418011.is3107.{table_name}`"
    return client.query(query).to_dataframe()


# db_manager = DatabaseManager(database_name="is3107")
# data = db_manager.read_table(table_name="housing_data")
data = pd.read_csv('testing/resale_flat_prices_2017-2024_new.csv')

conn = st.connection('gcs', type=FilesConnection)
data2 = conn.read("is3107_bucket/resale_price.csv", input_format="csv", ttl=600)

# Streamlit UI
st.title('IS3107 Project')

tab1, tab2, tab3 = st.tabs(["Main Predicted Resale Price", "Similar Properties", "Testing"])

init_streamlit_comm()

with st.sidebar:
    st.header("Input Parameters")
    # Numeric inputs
    floor_area_sqm = st.number_input('Floor Area (sqm)', min_value=0.0, format="%.2f")
    age_of_flat = st.number_input('Age of Flat', min_value=0.0, format="%.2f")
    avg_storey_range = st.number_input('Average Storey Range', min_value=0.0, format="%.2f")
    total_dwelling_units = st.number_input('Total Dwelling Units', min_value=0, format="%d")

    # Boolean input
    commercial = st.checkbox('Commercial')

    # Dropdown inputs
    flat_type = st.selectbox('Flat Type', ['3 room', '4 room', '5 room', 'other'])
    flat_model = st.selectbox('Flat Model', [
        'apartment', 'improved', 'maisonette', 'model a', 'new generation',
        'other', 'premium apartment', 'simplified', 'standard'
    ])

    # Submit button
    submit_button = st.button("Submit")

with tab1:
    col1, col2, col3 = st.columns([1,3, 1])
    with col2:
        if submit_button:
            st.header("Selected Parameters")
            st.write(f"Floor Area (sqm): {floor_area_sqm}")
            st.write(f"Age of Flat: {age_of_flat}")
            st.write(f"Average Storey Range: {avg_storey_range}")
            st.write(f"Commercial: {commercial}")
            st.write(f"Total Dwelling Units: {total_dwelling_units}")
            st.write(f"Flat Type: {flat_type}")
            st.write(f"Flat Model: {flat_model}")
            st.write(" ")
            st.header("Predicted Resale Price")
            st.write(f"Predicted Resale Price for your selection:")
            st.write(f"Empty for now")
            st.write(" ")
            st.header("Comparison")
            st.write(f"Average Resale Price for housing of {flat_type} and {flat_model}")
        else:
            st.write("Please input the required parameters")
            st.write("Then Press Submit")

with tab2:
    st.write("Testing:")
    st.dataframe(data2.head())

with tab3:
    st.header("Predict Price")
    # Input for prediction
    user_input = st.number_input('Enter a value for prediction:', min_value=0.0, format="%.1f")

    # Button for prediction
    predict_button = st.button("Predict")

    # When the user clicks 'Predict'
    if predict_button:
        # Load your model (consider loading it outside the function for efficiency)
        model = joblib.load("trained_model/random_forest_model.pkl")

        # Perform prediction
        predicted_price = prediction_price(user_input, model)

        # Display the predicted price
        st.write(f"Predicted Price: {predicted_price}")
    else:
        st.write("Press Predict button to predict")
# with tab2:
#     col1, col2, col3 = st.columns(3)
#     with col2:
#         st.header("Exploratory Data Analysis")
#         df = data.copy()
#
#     col4, col5 = st.columns(2)
#
#
#     def calculate_remaining_lease_years(lease):
#         parts = lease.split(' ')  # Split the string by spaces
#         years = int(parts[0])  # The first part is always the number of years
#
#         # If there are more than 2 parts, it means there's also a month component
#         if len(parts) > 2:
#             months = int(parts[2])  # The third part (index 2) is the number of months
#         else:
#             months = 0  # If no month component, set months to 0
#
#         # Convert the total lease to years, including months as a fractional year
#         total_years = years + months / 12.0
#         return total_years
#
#
#     # Apply the function to each row in the 'remaining_lease' column to create a new column
#     df['remaining_lease_years'] = df['remaining_lease'].apply(calculate_remaining_lease_years)
#     with col4:
#         # Trend Over Time
#         data['year_month_str'] = data['month'].astype(str)
#         average_prices_over_time = data.groupby('year_month_str')['resale_price'].mean().reset_index()
#         fig = px.line(average_prices_over_time, x='year_month_str', y='resale_price',
#                       title='Average Resale Prices Over Time')
#         st.plotly_chart(fig)
#         # Average resale price by town
#         price_by_town = df.groupby('town')['resale_price'].mean().sort_values(ascending=False).reset_index()
#         fig3 = px.bar(price_by_town, x='resale_price', y='town', orientation='h', title='Average Resale Price by Town')
#         st.plotly_chart(fig3)
#
#         # Average resale price by flat type
#         price_by_flat_type = df.groupby('flat_type')['resale_price'].mean().sort_values(ascending=False).reset_index()
#         fig4 = px.bar(price_by_flat_type, x='flat_type', y='resale_price', title='Average Resale Price by Flat Type')
#         st.plotly_chart(fig4)
#
#         st.write("This is a random chart")
#         map_data = pd.DataFrame(
#             np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#             columns=['lat', 'lon'])
#
#         st.map(map_data)
#     with col5:
#         fig5 = px.scatter(df, x='floor_area_sqm', y='resale_price', title='Resale Price vs. Floor Area',
#                           trendline="ols",
#                           opacity=0.5)
#         st.plotly_chart(fig5)
#
#         fig6 = px.scatter(df, x='remaining_lease_years', y='resale_price',
#                           title='Resale Price vs. Remaining Lease Years',
#                           opacity=0.5)
#         st.plotly_chart(fig6)
#
#         # Calculating the correlation matrix
#         correlation_matrix = df[
#             ['floor_area_sqm', 'lease_commence_date', 'resale_price', 'remaining_lease_years']].corr()
#
#         # Using Plotly's Figure Factory to create the heatmap
#         fig7 = ff.create_annotated_heatmap(
#             z=correlation_matrix.values,
#             x=list(correlation_matrix.columns),
#             y=list(correlation_matrix.index),
#             annotation_text=correlation_matrix.round(2).values,
#             showscale=True,
#             colorscale='Viridis'
#         )
#
#         fig7.update_layout(title_text='Correlation Matrix', xaxis_title="Features", yaxis_title="Features")
#
#         # Display the plot in Streamlit
#         st.plotly_chart(fig7)

# with tab3:
#     st.header("Tableau Style EDA")
#     st.write("This can be our EDA Tab ")
#
#
#     @st.cache_resource
#     def get_pyg_renderer() -> "StreamlitRenderer":
#         return StreamlitRenderer(data, spec="./gw_config.json", debug=False)
#
#
#     renderer = get_pyg_renderer()
#     renderer.render_explore()


# Note: The plotting and model logic is simplified and needs to be adapted based on your actual model and data.
