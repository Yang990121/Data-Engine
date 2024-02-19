# layout.py

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from data_sets.testing import create_dataset
from utils.graphing import render_graph_col

df = create_dataset()
temp_df = dash_table.DataTable(id="temp-df",
                               # columns=df.columns,
                               data=df.to_dict("records"))

layout = html.Div([
        dbc.Tabs(
            active_tab="testing",
            children=[
                dbc.Tab(label="Testing", id="testing", style={'marginTop': '20px'},
                        children=[
                            dbc.Row(
                                children=[
                                    dbc.Col(dcc.Dropdown(
                                        id='column-dropdown',
                                        options=[{'label': i, 'value': i} for i in df.columns],
                                        value='Y')),
                                ]
                            ),
                            dbc.Row(
                                children=[
                                    dcc.Graph(id='prediction-graph')
                                ]
                            ),
                            dbc.Row(
                                children=[
                                    dbc.Col(children=[
                                        dbc.Spinner(temp_df, type="grow")
                                    ])
                                ]
                            ),


                        ])  # Add margin at the top of the tab
                    ]
                )
            ], style={'margin': '10px'})  # Add margin around the entire div
