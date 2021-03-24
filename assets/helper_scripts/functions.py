import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

df = pd.read_csv(
    'https://raw.githubusercontent.com/dreth/UC3MStatisticalLearning/main/data/without_tags/data.csv')

# %% OTHER FUNCTIONS
# correlation table function
def corr_table(dataset, var1, var2):
    methods = ['pearson', 'kendall', 'spearman']
    corrs = [dataset.corr(method=x).loc[var1, var2] for x in methods]
    corr_table = {
        'coefficient': methods,
        'correlation': corrs
    }
    return pd.DataFrame(corr_table)

# correlation matrix heatmap function
def corr_matrix_max(dataset, methods=['pearson', 'kendall', 'spearman']):
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

# sort values on dataframe according to var
def sort_var(dataset, var, n=10, bottom_to_top=False):
    result = dataset.sort_values([var], axis='index', ascending=bottom_to_top)
    result = result.iloc[1:n,:]
    return result.reset_index(drop=True)

# %% PLOTTING FUNCTIONS
# barplot function
def bar(dataset, x, y, groupvar=False):
    params = {
        'data_frame': dataset,
        'x': x,
        'y': y
    }
    if groupvar != False:
        params['color'] = groupvar
        params['category_orders'] = {x:dataset[x]}
    return px.bar(**params)

# histogram function
def hist(dataset, x, groupvar=False, nbins=False):
    params = {
        'data_frame': dataset,
        'x': x
    }
    if groupvar != False:
        params['color'] = groupvar
    if nbins != False:
        params['nbins'] = nbins
    return px.histogram(**params)

# boxplot function
def box(dataset, x, y=False, horiz=True, groupvar=False):
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

# scatterplot function
def scatter(dataset, x, y, flip=False, groupvar=False, size=False):
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
    return px.scatter(**params)

# function to plot the corr matrix heatmap
def corr_matrix_heatmap(dataset):
    mat = corr_matrix_max(dataset)['corrs']
    return px.imshow(mat.values,
                     x=mat.columns,
                     y=mat.columns)

# function to plot top n for a variable with n entries
def topn(dataset, var, indexer='country_code', n=10,  bottom_to_top=False, groupvar=False):
    result = sort_var(dataset, var, n=n, bottom_to_top=bottom_to_top)
    return bar(result, indexer, var, groupvar=groupvar)
