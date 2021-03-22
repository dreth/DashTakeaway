import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px

df = pd.read_csv(
    'https://raw.githubusercontent.com/dreth/UC3MStatisticalLearning/main/data/without_tags/data.csv')

# barplot function
def bar(dataset, x, y, groupvar=False):
    params = {
        'data_frame': dataset,
        'x': x,
        'y': y
    }
    if groupvar != False:
        params['color'] = groupvar
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
        params['color'] = y
        params['y'] = y
    return px.box(**params)

