import plotly.graph_objects as go
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from .plot_helper import graph_config

def render_graph(graph_id, fig: go.Figure):
    return dcc.Graph(
        id=graph_id,
        figure=fig,
        config=graph_config
    )
def render_graph_col(col_header: str, graph_id: str, fig: go.Figure,
                     tooltip_id: str = None, tooltip_msg: str=None):
    if tooltip_id and tooltip_msg:
        return dbc.Col(children=[
            html.H4(col_header, style={"display": "inline-block"}),
            html.Span(html.I(className="fas fa-question-circle"), id=tooltip_id),
            dbc.Tooltip(tooltip_msg, target=tooltip_id),
            dbc.Spinner([render_graph(graph_id, fig)], type='grow')
            ]
        )
    else:
        return dbc.Col(children=[
            html.H4(col_header),
            dbc.Spinner([render_graph(graph_id, fig)], type="grow")
        ])