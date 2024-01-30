import flask
import dash
import dash_bootstrap_components as dbc

from layout import layout

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.ZEPHYR, dbc.icons.BOOTSTRAP, dbc.icons.FONT_AWESOME],
                suppress_callback_exceptions=True,
                prevent_initial_callbacks = "initial_duplicate"
                )

app.server.secret_key = "testkey"

app.title = "IS3107 Project"

app.layout = layout

server = app.server