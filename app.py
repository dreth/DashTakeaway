
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

# Bootstrap stylesheet
external_stylesheets = [
    'https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/cyborg/bootstrap.min.css']

# Dash app initialization
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Sidebar style arguments
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#000000",
    "color": "#FFFFFF",
    "font-family": "ubuntu, sans-serif"
}

# Content style arguments
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem"
}

# Sidebar
sidebar = html.Div(
    [
        html.H3("The development dataset", className="display-6"),
        html.Hr(),
        # Navbar with main links
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact",
                            external_link=True),
                dbc.NavLink("Page 1", href="/page-1",
                            active="exact", external_link=True),
                dbc.NavLink("Page 2", href="/page-2",
                            active="exact", external_link=True),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.P("This is the content of the home page!")
    elif pathname == "/page-1":
        return html.P("This is the content of page 1. Yay!")
    elif pathname == "/page-2":
        return html.P("Oh cool, this is page 2!")

    # 404
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The path {pathname} does not exist"),
        ]
    )


if __name__ == "__main__":
    app.run_server(port=5665)
