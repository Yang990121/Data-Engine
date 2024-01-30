# layout.py

from dash import html, dcc
import dash_bootstrap_components as dbc
from data_sets.testing import create_dataset

df = create_dataset()

layout = html.Div([
        dbc.Tabs(
            active_tab="testing",
            children=[
                dbc.Tab(label="Testing", children=[
                    dcc.Dropdown(
                        id='column-dropdown',
                        options=[{'label': i, 'value': i} for i in df.columns],
                        value='Y'  # Default value
                    ),
                    dcc.Graph(id='prediction-graph'),  # Graph component to display the plot
                ], id="testing", style={'marginTop': '20px'})  # Add margin at the top of the tab
            ]
        )
    ], style={'margin': '10px'})  # Add margin around the entire div
