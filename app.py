
from os import name
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
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, title='Multiple dataset Dash app')

# %% LAYOUT STYLES

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

# %% APP LAYOUT

# Iterative generator of page navlinks for development dataset
sidebar_tabs_dev = ['Development dataset', 'Histograms',
                    'Boxplots', 'Correlation', 'HDI Aggregates', 'Top N countries']


# Iterative generator of page navlinks for ramen ratings dataset
sidebar_tabs_ram = ['Ramen ratings', 'Top 10 by Rating',
                    'Top 10 with details', 'Regression for Ratings']

# function to convert sidebar tab names into sidebar tabs


def sidebar_tabs(tabnames, active='exact', external_link=True):
    navlinks = [0 for x in tabnames]
    for i in range(len(tabnames)):
        # params
        tabname = tabnames[i]
        href_add = f"/{tabname.lower().replace(' ','-')}"

        # generating the list of navlinks
        navlink = dbc.NavLink(tabname.lower(),
                                href=href_add,
                                active=active,
                                external_link=external_link)
        navlinks[i] = navlink
    return navlinks


# Sidebar
sidebar = html.Div(
    [
        html.H3("Dash App", className="fs-4", id='sideBarTitle'),
        html.Hr(),
        html.H5("The development dataset", className="fs-5 sideBarDatasetTitles"),
        # Navbar with main links
        dbc.Nav(
            sidebar_tabs(sidebar_tabs_dev),
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.H5("Ramen ratings", className="fs-5 sideBarDatasetTitles"),
        dbc.Nav(
            sidebar_tabs(sidebar_tabs_ram),
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# content div
content = html.Div(id="page-content", style=CONTENT_STYLE)

# app general layout
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# %% PAGE CONTENT


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):

    # root directory / homepage
    if pathname == '/':
        return dbc.Jumbotron([
            html.H2("Welcome to our Dash App!", id='mainTitle'),
            html.P("The developers of this app are:"),
            html.Ul([
                html.Li(html.A("Danyu Zhang",href='https://github.com/danyuz')),
                html.Li(html.A("Daniel Alonso",href='https://github.com/dreth'))
            ]),
            html.P("This application uses 2 datasets, specific information about each dataset is detailed in the following links:"),
            html.Ul([
                html.Li(html.A("The development dataset",href='/The-dataset')),
                html.Li(html.A("Ramen ratings dataset",href='/Introduction'))
            ]),
            html.P("Feel free to navigate the page and play around with the menus!")
        ])

    # DEVELOPMENT DATASET PAGES
    elif pathname == "/development-dataset":
        return html.Div([
            html.H3(
                "A set of visualizations for different worldwide demographic/development metrics")])


    # RAMEN DATASET PAGES
    elif pathname == "/ramen-ratings":
        return html.P("This is the content of the home page!")

    
    # 404 ERROR MESSAGE PAGE
    else:
        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The path {pathname} does not exist"),
            ]
        )

# %% SERVER


# run server
if __name__ == "__main__":
    app.run_server(port=5665)
