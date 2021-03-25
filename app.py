from os import name
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import dash_table
from dash.dependencies import Input, Output
import json

# Bootstrap stylesheet
external_stylesheets = [
    'https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/cyborg/bootstrap.min.css']

# Dash app initialization
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, title="dan+dandan's Dash App")

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

# %% IMPORTING THE DATA
# DEVELOPMENT DATA
# importing data
dev_df = pd.read_csv(
    'https://raw.githubusercontent.com/dreth/UC3MStatisticalLearning/main/data/without_tags/data.csv')

# numeric cols
dev_val_cols = list(dev_df.columns[2:len(dev_df.columns)-1])
dev_val_col_options = [{'label': x, 'value': x}
                       for x in dev_val_cols]
dev_val_col_options_non_neg = [{'label': x, 'value': x}
                               for x in dev_val_cols if True not in list(dev_df[x] < 0)]

# identity cols
dev_identity_cols = list(dev_df.columns[0:2])
dev_id_col_options = [{'label': x, 'value': x}
                      for x in dev_identity_cols]

# grouping col
dev_group_col = 'hdi_cat'

# %% APP LAYOUT
# Iterative generator of page navlinks for development dataset
sidebar_tabs_dev = ['Development dataset', 'Histograms',
                    'Boxplots', 'Correlation', 'Top N countries']


# Iterative generator of page navlinks for ramen ratings dataset
sidebar_tabs_ram = ['Ramen ratings', 'Top 10 by Rating',
                    'Top 10 with details', 'Regression for Ratings']


def sidebar_tabs(tabnames, active='exact', external_link=True):
    # function to convert sidebar tab names into sidebar tabs
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


# SIDEBAR
sidebar = html.Div(
    [
        html.H3("Dash App", className="fs-4", id='sideBarTitle'),
        html.Hr(),
        html.H5("The development dataset",
                className="fs-5 sideBarDatasetTitles"),
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

# MAIN CONTENT
content = html.Div(id="page-content", style=CONTENT_STYLE)

# APP LAYOUT CALL
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# %% FUNCTIONS TO PLOT AND ORDER DATA


def corr_table(dataset, var1, var2):
    # correlation table function
    methods = ['pearson', 'kendall', 'spearman']
    corrs = [dataset.corr(method=x).loc[var1, var2] for x in methods]
    corr_table = {
        'coefficient': methods,
        'correlation': corrs
    }
    return pd.DataFrame(corr_table)


def corr_matrix_max(dataset, methods=['pearson', 'kendall', 'spearman']):
    # correlation matrix heatmap function
    # dict for corr matrices per method
    matrices = {}
    # empty df
    max_corr_df = dataset.corr()
    method_used = max_corr_df.copy()
    max_corr_df[:] = 0
    method_used[:] = 0
    # create a df per method
    for method in methods:
        matrices[method] = dataset.corr(method=method)
    # find max of each correl
    for i in range(len(max_corr_df)):
        for j in range(len(max_corr_df)):
            # correl per method
            method_vals = [matrices['pearson'].iloc[i, j],
                           matrices['kendall'].iloc[i, j],
                           matrices['spearman'].iloc[i, j]]
            # best correlation
            method_vals_abs = [abs(x) for x in method_vals]
            best_c = max(method_vals_abs)
            # adding to i,j the best correlation
            max_corr_df.iloc[i, j] = method_vals[method_vals_abs.index(best_c)]
            # appending what method was used to obtain previous correl
            method_used.iloc[i, j] = methods[method_vals_abs.index(best_c)]
    return {'corrs': max_corr_df, 'methods': method_used}


def sort_var(dataset, var, n=10, bottom_to_top=False):
    # sort values on dataframe according to var
    result = dataset.sort_values([var], axis='index', ascending=bottom_to_top)
    result = result.iloc[1:n, :]
    return result.reset_index(drop=True)

# PLOTTING FUNCTIONS


def bar(dataset, x, y, groupvar=False):
    # barplot function
    params = {
        'data_frame': dataset,
        'x': x,
        'y': y
    }
    if groupvar != False:
        params['color'] = groupvar
        params['category_orders'] = {x: dataset[x]}
    return px.bar(**params)


def hist(dataset, x, groupvar=False, nbins=False):
    # histogram function
    params = {
        'data_frame': dataset,
        'x': x
    }
    if groupvar != False:
        params['color'] = groupvar
    if nbins != False:
        params['nbins'] = nbins
    return px.histogram(**params)


def box(dataset, x, y=False, horiz=True, groupvar=False):
    # boxplot function
    params = {
        'data_frame': dataset,
        'x': x
    }
    if y != False:
        params['y'] = y
    if horiz != True:
        params['y'] = x
        params['x'] = y
    if groupvar != False:
        params['color'] = groupvar
    return px.box(**params)


def scatter(dataset, x, y, flip=False, groupvar=False, size=False, hover_data=False):
    # scatterplot function
    params = {
        'data_frame': dataset,
        'x': x,
        'y': y
    }
    if flip != False:
        params['x'] = y
        params['y'] = x
    if groupvar != False:
        params['color'] = groupvar
    if size != False:
        params['size'] = size
    if hover_data != False:
        params['hover_data'] = hover_data
    return px.scatter(**params)


def corr_matrix_heatmap(dataset):
    # function to plot the corr matrix heatmap
    mat = corr_matrix_max(dataset)['corrs']
    return px.imshow(mat.values,
                     x=mat.columns,
                     y=mat.columns)


def topn(dataset, var, indexer='country_code', n=10,  bottom_to_top=False, groupvar=False):
    # function to plot top n for a variable with n entries
    result = sort_var(dataset, var, n=n, bottom_to_top=bottom_to_top)
    return bar(result, indexer, var, groupvar=groupvar)

# %% PAGE CONTENT
# website content layout and text


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    # HOMEPAGE
    if pathname == '/':
        return dbc.Jumbotron([
            html.H2("Welcome to our Dash App!", id='mainTitle'),
            html.P("The developers of this app are:"),
            html.Ul([
                html.Li(html.A("Danyu Zhang", href='https://github.com/danyuz')),
                html.Li(html.A("Daniel Alonso", href='https://github.com/dreth'))
            ]),
            html.P("This application uses 2 datasets, specific information about each dataset is detailed in the following links:"),
            html.Ul([
                html.Li(html.A("The development dataset", href='/The-dataset')),
                html.Li(html.A("Ramen ratings dataset", href='/Introduction'))
            ]),
            html.P("Feel free to navigate the page and play around with the menus!")
        ])

    ## DEVELOPMENT DATASET PAGES ##
    # development dataset introduction
    elif pathname == "/development-dataset":
        return dbc.Jumbotron([
            html.H3(
                "A set of visualizations for different worldwide demographic/development metrics"),
            html.P([
                "Obtained form the ",
                html.A("World bank Databank",
                       href="https://databank.worldbank.org/home.aspx"),
                "specifically the ",
                html.A("World Development Indicators database",
                       href="https://databank.worldbank.org/source/world-development-indicators"),
                "This is the 'primary World Bank collection of development indicators' as stated on the database description. It has lots of economic, education, energy use, and population specific metrics."
            ]),
            html.P([
                "The dataset used (as it was used) can be found in the ",
                html.A("file tree of dreth's Statistical Learning project",
                       href="https://github.com/dreth/UC3MStatisticalLearning/tree/main/data"),
                ". The file path is as follows: /data/without_tags/data.csv."
            ]),
            html.H4("Dataset Variables:"),
            html.Ul([
                html.Li([html.Strong("year_code"),
                         ": code for the year as the world bank databank sets it"]),
                html.Li([html.Strong("country_name"), ": name of the country"]),
                html.Li([html.Strong("country_code"),
                         ": alpha-3 ISO 3166 code for the country"]),
                html.Li([html.Strong("foreign_inv_inflows"),
                         ": Foreign direct investment, net inflows (BoP, current US$)"]),
                html.Li([html.Strong("exports_perc_gdp"),
                         ": Exports of goods and services (as a % of GDP)"]),
                html.Li([html.Strong("inflation_perc"),
                         ": Inflation, consumer prices (annual %)"]),
                html.Li([html.Strong("education_years"),
                         ": Compulsory education, duration (years)"]),
                html.Li([html.Strong("education_perc_gdp"),
                         ": Government expenditure on education, total (as a % of GDP)"]),
                html.Li([html.Strong("gds_perc_gdp"),
                         ": Gross domestic savings (as a % of GDP)"]),
                html.Li([html.Strong("gross_savings_perc_gdp"),
                         ": Gross savings (as a % of GDP)"]),
                html.Li([html.Strong("int_tourism_arrivals"),
                         ": International tourism, number of arrivals"]),
                html.Li([html.Strong("int_tourism_receipts"),
                         ": International tourism, receipts (in current US$)"]),
                html.Li([html.Strong("perc_internet_users"),
                         ": Individuals using the Internet (as a % of population)"]),
                html.Li([html.Strong("access_to_electricity"),
                         ": Access to electricity (% of population)"]),
                html.Li([html.Strong("agricultural_land"),
                         ": Agricultural land (% of land area)"]),
                html.Li([html.Strong("birth_rate"),
                         ": Birth rate, crude (per 1,000 people)"]),
                html.Li(
                    [html.Strong("gne"), ": Gross national expenditure (% of GDP)"]),
                html.Li([html.Strong("mobile_subscriptions"),
                         ": Mobile cellular subscriptions (per 100 people)"]),
                html.Li([html.Strong("infant_mort_rate"),
                         ": Mortality rate, infant (per 1,000 live births)"]),
                html.Li([html.Strong("sex_ratio"),
                         ": Sex ratio at birth (male births per female births)"]),
                html.Li([html.Strong("greenhouse_gas_em"),
                         ": Total greenhouse gas emissions (kt of CO2 equivalent)"]),
                html.Li([html.Strong("urban_pop_perc"),
                         ": Urban population (% of total population)"]),
                html.Li([html.Strong("hdi"), ": human development index "]),
                html.Li([html.Strong("hdi_cat"),
                         ": Human development index as a category"]),
                html.Li([html.Strong("life_exp"),
                         ": Life expectancy at birth, total (years)"]),
                html.Li([html.Strong("gdp"), ": GDP (current US$) "]),
                html.Li([html.Strong("gni"), ": GNI (current US$)"]),
                html.Li([html.Strong("fertility_rate"),
                         ": Fertility rate, total (births per woman)"])
            ]),
            html.Img(src="https://raw.githubusercontent.com/dreth/UC3MDataTidyingAndReporting/main/First-takeaway/www/worldbanklogo.png", width=462.222, height=260)
        ])

    # Histogram section
    elif pathname == '/histograms':
        return html.Div([
            html.H3('Histograms per variable'),
            dbc.Jumbotron([
                html.Label('Select variable'),
                dcc.Dropdown(id='variableSelectorHist',
                             options=dev_val_col_options,
                             value=np.random.choice(dev_val_cols)
                             ),
                html.Br(),
                html.Label('HDI grouping'),
                dcc.RadioItems(id='groupByHDIHist',
                               options=[
                                  {'label': 'Group by HDI', 'value': 1},
                                  {'label': 'Do not group', 'value': 0}
                               ],
                               value=1,
                               labelStyle={'display': 'block'}
                               )
            ]),
            dbc.Jumbotron([
                dcc.Graph(id="devHistOutput")
            ])
        ])

    # Boxplot section
    elif pathname == '/boxplots':
        return html.Div([
            dbc.Jumbotron([
                html.H3('Boxplot per variable'),
                html.Label('Select variable'),
                dcc.Dropdown(id='variableSelectorBox',
                             options=dev_val_col_options,
                             value=np.random.choice(dev_val_cols)
                             ),
                html.Br(),
                html.Label('HDI grouping'),
                dcc.RadioItems(id='groupByHDIBox',
                               options=[
                                  {'label': 'Group by HDI', 'value': 1},
                                  {'label': 'Do not group', 'value': 0}
                               ],
                               value=1,
                               labelStyle={'display': 'block'}
                               )
            ]),
            dbc.Jumbotron([
                dcc.Graph(id="devBoxOutput")
            ])
        ])

    # Correlation section
    elif pathname == '/correlation':
        return html.Div([
            html.H4('Scatter plot + Correlation between variables'),
            dbc.Jumbotron([
                html.Label('Select variable 1'),
                dcc.Dropdown(id='variableSelectorScatter1',
                             options=dev_val_col_options,
                             value=np.random.choice(dev_val_cols)
                             ),
                html.Br(),
                html.Label('Select variable 2'),
                dcc.Dropdown(id='variableSelectorScatter2',
                             options=dev_val_col_options,
                             value=np.random.choice(dev_val_cols)
                             ),
                html.Br(),
                html.Label('Select variable for dot size'),
                dcc.Dropdown(id='variableSelectorScatter3',
                             options=[{'label': 'None', 'value': 'None'}
                                      ] + dev_val_col_options_non_neg,
                             value='None'
                             ),
                html.Br(),
                html.Label('Select variable to identify dots'),
                dcc.Dropdown(id='variableSelectorScatterHover',
                             options=dev_id_col_options,
                             value='country_name'
                             ),
                html.Br(),
                html.Label('HDI grouping'),
                dcc.RadioItems(id='groupByHDIScatter',
                               options=[
                                   {'label': 'Group by HDI', 'value': 1},
                                   {'label': 'Do not group', 'value': 0}
                               ],
                               value=1,
                               labelStyle={'display': 'block'}
                               )
            ]),
            dbc.Jumbotron([
                html.Label('Correlation coefficient computations for the 2 selected variables'),
                html.Div(id='correlDataTable')
            ]),
            dbc.Jumbotron([
                html.Label('Scatter plot result'),
                dcc.Graph(id="devScatterOutput")
            ]),
            html.Div(id="correlationOutputData")
        ])

    # top n countries section
    elif pathname == '/top-n-countries':
        return html.Div([
            html.H4('Visualizing top N countries per variable'),
            dbc.Jumbotron([
                html.Label('Select variable'),
                dcc.Dropdown(id='variableSelectorTopN',
                             options=dev_val_col_options,
                             value=np.random.choice(dev_val_cols)
                             ),
                html.Br(),
                html.Label('Select id variable'),
                dcc.Dropdown(id='variableSelectorTopNId',
                             options=dev_id_col_options,
                             value=np.random.choice(dev_identity_cols)
                             ),
                html.Br(),
                html.Label('Amount of countries to show'),
                dcc.Slider(id='amountCountriesTopNSlider',
                           min=1,
                           max=184,
                           value=10,
                           marks={**{1: '1', 184: '184'}, **
                                  {x: str(x) for x in range(1, 184) if x % 21 == 0}},
                           step=1
                           ),
                html.Br(),
                html.Br(),
                html.Label('Country sorting'),
                dcc.RadioItems(id='sortingTopN',
                               options=[
                                   {'label': 'Descending', 'value': 0},
                                   {'label': 'Ascending', 'value': 1}
                               ],
                               value=0,
                               labelStyle={'display': 'block'}
                               ),
                html.Br(),
                html.Label('HDI grouping'),
                dcc.RadioItems(id='groupByHDITopN',
                               options=[
                                   {'label': 'Group by HDI', 'value': 1},
                                   {'label': 'Do not group', 'value': 0}
                               ],
                               value=1,
                               labelStyle={'display': 'block'}
                               ),

            ]),
            dbc.Jumbotron([
                dcc.Graph(id="devTopNOutput")
            ])
        ])

    # RAMEN DATASET PAGES
    # ramen ratings dataset introduction
    elif pathname == "/ramen-ratings":
        return html.P("This is the content of the home page!")

    # 404 ERROR MESSAGE PAGE
    else:
        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(["The path ",
                        html.Span(f"{pathname}", id='pathName'),
                        " does not exist"]),
            ]
        )

# %% DEVELOPMENT PLOT CALLBACKS
# Histograms
@app.callback(Output('devHistOutput', 'figure'),
              [Input('variableSelectorHist', 'value'),
              Input('groupByHDIHist', 'value')])
def update_graph(variableSelectorHist, groupByHDIHist):
    if groupByHDIHist == 1:
        groupByHDIHist = 'hdi_cat'
    else:
        groupByHDIHist = False
    return hist(dev_df, x=variableSelectorHist, groupvar=groupByHDIHist)

# Boxplots
@app.callback(Output('devBoxOutput', 'figure'),
              [Input('variableSelectorBox', 'value'),
              Input('groupByHDIBox', 'value')])
def update_graph(variableSelectorBox, groupByHDIBox):
    if groupByHDIBox == 1:
        groupByHDIBox = 'hdi_cat'
    else:
        groupByHDIBox = False
    return box(dev_df, x=variableSelectorBox, groupvar=groupByHDIBox)

# Correlation
@app.callback(Output('devScatterOutput', 'figure'),
              [Input('variableSelectorScatter1', 'value'),
              Input('variableSelectorScatter2', 'value'),
              Input('variableSelectorScatter3', 'value'),
              Input('variableSelectorScatterHover', 'value'),
              Input('groupByHDIScatter', 'value')])
def update_graph(variableSelectorScatter1, variableSelectorScatter2, variableSelectorScatter3, variableSelectorScatterHover, groupByHDIScatter):
    if variableSelectorScatter3 == 'None':
        variableSelectorScatter3 = False
    if groupByHDIScatter == 1:
        groupByHDIScatter = 'hdi_cat'
    else:
        groupByHDIScatter = False
    return scatter(dataset=dev_df, x=variableSelectorScatter1, y=variableSelectorScatter2, size=variableSelectorScatter3, groupvar=groupByHDIScatter, hover_data=[variableSelectorScatterHover])

# Correlation coefs
@app.callback(Output('correlDataTable', 'children'),
              [Input('variableSelectorScatter1', 'value'),
              Input('variableSelectorScatter2', 'value')])
def update_graph(variableSelectorScatter1, variableSelectorScatter2):
    data = corr_table(dev_df, var1=variableSelectorScatter1,
                      var2=variableSelectorScatter2)
    table = dash_table.DataTable(
        columns=[{'name': x, 'id': x} for x in data.columns], data=data.to_dict('records'),
        style_cell={'color':'white',
        'backgroundColor':'black'},
        style_header={'color':'aquamarine',
        'backgroundColor':'black'})
    return table

# Top N
@app.callback(Output('devTopNOutput', 'figure'),
              [Input('variableSelectorTopN', 'value'),
              Input('variableSelectorTopNId', 'value'),
              Input('amountCountriesTopNSlider', 'value'),
              Input('groupByHDITopN', 'value'),
              Input('sortingTopN', 'value')])
def update_graph(variableSelectorTopN, variableSelectorTopNId, amountCountriesTopNSlider, groupByHDITopN, sortingTopN):
    if groupByHDITopN == 1:
        groupByHDITopN = 'hdi_cat'
    else:
        groupByHDITopN = False
    if sortingTopN == 1:
        sortingTopN = True
    else:
        sortingTopN = False
    return topn(dataset=dev_df, var=variableSelectorTopN, indexer=variableSelectorTopNId, n=amountCountriesTopNSlider, groupvar=groupByHDITopN, bottom_to_top=sortingTopN)

# Top N bottom table from plot selection
@app.callback(Output('correlationOutputData','children'),
              [Input('devScatterOutput','selectedData'),
              Input('variableSelectorScatter1', 'value'),
              Input('variableSelectorScatter2', 'value'),
              Input('variableSelectorScatter3', 'value'),
              Input('variableSelectorScatterHover', 'value')])
def display_table(devScatterOutput, variableSelectorScatter1, variableSelectorScatter2, variableSelectorScatter3, variableSelectorScatterHover):
    if devScatterOutput is None or len(devScatterOutput) == 0:
        return None
    cols = [
            {'name':variableSelectorScatterHover,'id':variableSelectorScatterHover},
            {'name':variableSelectorScatter1,'id':variableSelectorScatter1},
            {'name':variableSelectorScatter2,'id':variableSelectorScatter2}
    ]
    if variableSelectorScatter3 != 'None':
        cols.update({'name':variableSelectorScatter3,'id':variableSelectorScatter3})
    # finding out which countries are selected to filter
    countries = [o['customdata'][0] for o in devScatterOutput['points']]
    # filtering the data
    filtered_df_records = dev_df[dev_df[variableSelectorScatterHover].isin(countries)].to_dict('records') 
    # table with data filtered
    tbl = dash_table.DataTable(columns=cols,
                              data=filtered_df_records,
                              style_cell={'color':'white',
                              'backgroundColor':'black'},
                              style_header={'color':'aquamarine',
                              'backgroundColor':'black'})
    return dbc.Jumbotron([html.Label('Table for data selected in the scatterplot'),tbl])
                               

# %% RAMEN RATINGS PLOT CALLBACKS

# %% SERVER
# run server
if __name__ == "__main__":
    app.run_server(port=5665, debug=True)
