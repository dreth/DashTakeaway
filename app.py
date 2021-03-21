
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

# %% PLOTTING FUNCTIONS
# functions used to plot each element in each tab of the app




# %% PAGE CONTENT
# website content layout and text
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):

    #### HOMEPAGE
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

    #### DEVELOPMENT DATASET PAGES
    # development dataset introduction
    elif pathname == "/development-dataset":
        return dbc.Jumbotron([
            html.H3("A set of visualizations for different worldwide demographic/development metrics"),
            html.P([
                "Obtained form the ",
                html.A("World bank Databank", href="https://databank.worldbank.org/home.aspx"),
                "specifically the ",
                html.A("World Development Indicators database", href="https://databank.worldbank.org/source/world-development-indicators"),
                "This is the 'primary World Bank collection of development indicators' as stated on the database description. It has lots of economic, education, energy use, and population specific metrics."
            ]),
            html.P([
                "The dataset used (as it was used) can be found in the ",
                html.A("file tree of dreth's Statistical Learning project", href="https://github.com/dreth/UC3MStatisticalLearning/tree/main/data"),
                ". The file path is as follows: /data/without_tags/data.csv."
            ]),
            html.H4("Dataset Variables:"),
            html.Ul([
                html.Li([html.Strong("year_code"), ": code for the year as the world bank databank sets it"]),
                html.Li([html.Strong("country_name"), ": name of the country"]),
                html.Li([html.Strong("country_code"), ": alpha-3 ISO 3166 code for the country"]),
                html.Li([html.Strong("foreign_inv_inflows"), ": Foreign direct investment, net inflows (BoP, current US$)"]),
                html.Li([html.Strong("exports_perc_gdp"), ": Exports of goods and services (as a % of GDP)"]),
                html.Li([html.Strong("inflation_perc"), ": Inflation, consumer prices (annual %)"]),
                html.Li([html.Strong("education_years"), ": Compulsory education, duration (years)"]),
                html.Li([html.Strong("education_perc_gdp"), ": Government expenditure on education, total (as a % of GDP)"]),
                html.Li([html.Strong("gds_perc_gdp"), ": Gross domestic savings (as a % of GDP)"]),
                html.Li([html.Strong("gross_savings_perc_gdp"), ": Gross savings (as a % of GDP)"]),
                html.Li([html.Strong("int_tourism_arrivals"), ": International tourism, number of arrivals"]),
                html.Li([html.Strong("int_tourism_receipts"), ": International tourism, receipts (in current US$)"]),
                html.Li([html.Strong("perc_internet_users"), ": Individuals using the Internet (as a % of population)"]),
                html.Li([html.Strong("access_to_electricity"), ": Access to electricity (% of population)"]),
                html.Li([html.Strong("agricultural_land"), ": Agricultural land (% of land area)"]),
                html.Li([html.Strong("birth_rate"), ": Birth rate, crude (per 1,000 people)"]),
                html.Li([html.Strong("gne"), ": Gross national expenditure (% of GDP)"]),
                html.Li([html.Strong("mobile_subscriptions"), ": Mobile cellular subscriptions (per 100 people)"]),
                html.Li([html.Strong("infant_mort_rate"), ": Mortality rate, infant (per 1,000 live births)"]),
                html.Li([html.Strong("sex_ratio"), ": Sex ratio at birth (male births per female births)"]),
                html.Li([html.Strong("greenhouse_gas_em"), ": Total greenhouse gas emissions (kt of CO2 equivalent)"]),
                html.Li([html.Strong("urban_pop_perc"), ": Urban population (% of total population)"]),
                html.Li([html.Strong("hdi"), ": human development index "]),
                html.Li([html.Strong("hdi_cat"), ": Human development index as a category"]),
                html.Li([html.Strong("life_exp"), ": Life expectancy at birth, total (years)"]),
                html.Li([html.Strong("gdp"), ": GDP (current US$) "]),
                html.Li([html.Strong("gni"), ": GNI (current US$)"]),
                html.Li([html.Strong("fertility_rate"), ": Fertility rate, total (births per woman)"])
            ]),
            html.Img(src="https://raw.githubusercontent.com/dreth/UC3MDataTidyingAndReporting/main/First-takeaway/www/worldbanklogo.png", width=462.222, height=260)
            ])


    #### RAMEN DATASET PAGES
    # ramen ratings dataset introduction
    elif pathname == "/ramen-ratings":
        return html.P("This is the content of the home page!")

    
    #### 404 ERROR MESSAGE PAGE
    else:
        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(["The path ",
                html.Span(f"{pathname}",id='pathName'),
                " does not exist"]),
            ]
        )

# %% SERVER

# run server
if __name__ == "__main__":
    app.run_server(port=5665)
