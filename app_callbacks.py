# app_callbacks.py

from dash.dependencies import Input, Output
from app import app
from models.testing_linear_regression import SimpleLinearRegressionModel
from data_sets.testing import create_dataset
import plotly.graph_objs as go

# Initialize the model with the dataset
df = create_dataset()
model = SimpleLinearRegressionModel(df)

@app.callback(
    Output('prediction-graph', 'figure'),  # Update this to output a figure
    Input('column-dropdown', 'value')
)
def update_graph(selected_column):
    # Prepare the data for plotting
    X = df.drop(columns=[selected_column])
    y = df[selected_column]
    predictions = model.predict(selected_column)

    # Create a scatter plot with the actual vs. predicted values
    figure = {
        'data': [
            go.Scatter(
                x=y,
                y=predictions,
                mode='markers',
                name='Predicted vs Actual'
            ),
            go.Scatter(
                x=[min(y), max(y)],
                y=[min(y), max(y)],
                mode='lines',
                name='Perfect Fit'
            )
        ],
        'layout': go.Layout(
            title='Linear Regression Predictions',
            xaxis={'title': 'Actual Values'},
            yaxis={'title': 'Predicted Values'},
            hovermode='closest'
        )
    }
    return figure  # Return the figure to be plotted
