# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import random
random.seed(123)

df = pd.read_csv("cleandata.csv")

# Allows for the labelling of plot.ly datapoints when hovering over them
players = ((df.player + "  " + df.pos).tolist())

# Principal Component Analysis with all components kept, used on all columns
# kept that may have an influence on determining a player's position
pca = PCA()

df_pca = pca.fit(df.iloc[:,4:37]).transform(df.iloc[:,4:37])

# Plot the players based on the values of their three most principal components
# Further visualization changes seen in the report were made in plot.ly's interface 
import plotly.plotly as py
import plotly.graph_objs as go
# Take only the 3 most principal components
df_pca3D = df_pca[:, 0:3]
x = df_pca3D[:,0]
y = df_pca3D[:,1]
z = df_pca3D[:,2]
py.sign_in('nthuy', 'upwufOB7qO0LmdxqEWo6')

trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    text=players,
    mode='markers',
    marker=dict(
        size=12,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)
#data = [trace1, trace2]
data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)