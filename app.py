import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go


# Simulated model and data set functions for demonstration
# Replace these with your actual functions from models/testing_linear_regression and data_sets/testing
def create_dataset():
    np.random.seed(0)
    X1 = np.random.randn(100)
    X2 = np.random.randn(100) * 2
    Y = X1 * 3 + X2 * 1.5 + np.random.randn(100)
    return pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})


class SimpleLinearRegressionModel:
    def __init__(self, df):
        self.df = df

    def predict(self, selected_column):
        # Dummy prediction logic
        if selected_column == 'X1':
            return self.df['X1'] * 3 + np.random.randn(len(self.df)) * 0.1
        elif selected_column == 'X2':
            return self.df['X2'] * 1.5 + np.random.randn(len(self.df)) * 0.1


# Load the dataset
df = create_dataset()

# Initialize the model
model = SimpleLinearRegressionModel(df)

# Streamlit UI
st.title('IS3107 Project')

tab1, tab2 = st.tabs(["Main", "Testing 123"])

with tab1:
    st.header("Main Tab")
    st.write("This can be our main prediction Tab")
    # Display the dataframe
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
            go.Scatter(x=[min(actual), max(actual)], y=[min(actual), max(actual)], mode='lines', name='Perfect Fit'))
        fig.update_layout(title='Linear Regression Predictions', xaxis_title='Actual Values',
                          yaxis_title='Predicted Values')
        st.plotly_chart(fig)

with tab2:
    st.header("Testing 123")
    st.write("This can be our EDA Tab ")

# Note: The plotting and model logic is simplified and needs to be adapted based on your actual model and data.
