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

st.set_page_config(layout="wide")

# Simulated model and data set functions for demonstration
# Replace these with your actual functions from models/testing_linear_regression and data_sets/testing
def create_dataset():
    np.random.seed(0)
    X1 = np.random.randn(100)
    X2 = np.random.randn(100) * 2
    Y = X1 * 3 + X2 * 1.5 + np.random.randn(100)
    return pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

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


class SimpleLinearRegressionModel:
    def __init__(self, df):
        self.df = df

    def predict(self, selected_column):
        # Dummy prediction logic
        if selected_column == 'X1':
            return self.df['X1'] * 3 + np.random.randn(len(self.df)) * 0.1
        elif selected_column == 'X2':
            return self.df['X2'] * 1.5 + np.random.randn(len(self.df)) * 0.1


# db_manager = DatabaseManager(database_name="is3107")
# data = db_manager.read_table(table_name="housing_data")
data = pd.read_csv('testing/resale_flat_prices_2017-2024_new.csv')

conn = st.connection('gcs', type=FilesConnection)
data2 = conn.read("is3107_bucket/resale_price.csv", input_format="csv", ttl=600)
# Load the dataset
df = create_dataset()

# Initialize the model
model = SimpleLinearRegressionModel(df)

# Streamlit UI
st.title('IS3107 Project')

tab1, tab2, tab3 = st.tabs(["Main", "EDA", "Tableau"])

init_streamlit_comm()

with tab1:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.header("Main Tab")
        st.write("This can be our main prediction Tab")
        # Display the dataframe
        st.write("Testing:")
        st.dataframe(data2.head())
        st.dataframe(query_table_from_bq("testing2"))
        # st.dataframe(data2)

        st.write("Dataframe:")
        st.dataframe(df)

        # Dropdown for selecting the prediction basis
        prediction_basis = st.selectbox('Predict with:', ['X1 vs Y', 'X2 vs Y'])

        if prediction_basis:
            selected_column = prediction_basis.split(' vs ')[0]
            # Predict
            predictions = model.predict(selected_column)

            # Actual values
            actual = df['Y']

            # Plotting the results
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=actual, y=predictions, mode='markers', name='Predicted vs Actual'))
            fig.add_trace(
                go.Scatter(x=[min(actual), max(actual)], y=[min(actual), max(actual)], mode='lines',
                           name='Perfect Fit'))
            fig.update_layout(title='Linear Regression Predictions', xaxis_title='Actual Values',
                              yaxis_title='Predicted Values')
            st.plotly_chart(fig)

with tab2:
    col1, col2, col3 = st.columns(3)
    with col2:
        st.header("Exploratory Data Analysis")
        df = data.copy()

    col4, col5 = st.columns(2)


    def calculate_remaining_lease_years(lease):
        parts = lease.split(' ')  # Split the string by spaces
        years = int(parts[0])  # The first part is always the number of years

        # If there are more than 2 parts, it means there's also a month component
        if len(parts) > 2:
            months = int(parts[2])  # The third part (index 2) is the number of months
        else:
            months = 0  # If no month component, set months to 0

        # Convert the total lease to years, including months as a fractional year
        total_years = years + months / 12.0
        return total_years


    # Apply the function to each row in the 'remaining_lease' column to create a new column
    df['remaining_lease_years'] = df['remaining_lease'].apply(calculate_remaining_lease_years)
    with col4:
        # Trend Over Time
        data['year_month_str'] = data['month'].astype(str)
        average_prices_over_time = data.groupby('year_month_str')['resale_price'].mean().reset_index()
        fig = px.line(average_prices_over_time, x='year_month_str', y='resale_price',
                      title='Average Resale Prices Over Time')
        st.plotly_chart(fig)
        # Average resale price by town
        price_by_town = df.groupby('town')['resale_price'].mean().sort_values(ascending=False).reset_index()
        fig3 = px.bar(price_by_town, x='resale_price', y='town', orientation='h', title='Average Resale Price by Town')
        st.plotly_chart(fig3)

        # Average resale price by flat type
        price_by_flat_type = df.groupby('flat_type')['resale_price'].mean().sort_values(ascending=False).reset_index()
        fig4 = px.bar(price_by_flat_type, x='flat_type', y='resale_price', title='Average Resale Price by Flat Type')
        st.plotly_chart(fig4)

        st.write("This is a random chart")
        map_data = pd.DataFrame(
            np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
            columns=['lat', 'lon'])

        st.map(map_data)
    with col5:
        fig5 = px.scatter(df, x='floor_area_sqm', y='resale_price', title='Resale Price vs. Floor Area',
                          trendline="ols",
                          opacity=0.5)
        st.plotly_chart(fig5)

        fig6 = px.scatter(df, x='remaining_lease_years', y='resale_price',
                          title='Resale Price vs. Remaining Lease Years',
                          opacity=0.5)
        st.plotly_chart(fig6)

        # Calculating the correlation matrix
        correlation_matrix = df[
            ['floor_area_sqm', 'lease_commence_date', 'resale_price', 'remaining_lease_years']].corr()

        # Using Plotly's Figure Factory to create the heatmap
        fig7 = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=list(correlation_matrix.columns),
            y=list(correlation_matrix.index),
            annotation_text=correlation_matrix.round(2).values,
            showscale=True,
            colorscale='Viridis'
        )

        fig7.update_layout(title_text='Correlation Matrix', xaxis_title="Features", yaxis_title="Features")

        # Display the plot in Streamlit
        st.plotly_chart(fig7)

with tab3:
    st.header("Tableau Style EDA")
    st.write("This can be our EDA Tab ")


    @st.cache_resource
    def get_pyg_renderer() -> "StreamlitRenderer":
        return StreamlitRenderer(data, spec="./gw_config.json", debug=False)


    renderer = get_pyg_renderer()
    renderer.render_explore()

    # st.dataframe(data)

# Note: The plotting and model logic is simplified and needs to be adapted based on your actual model and data.
