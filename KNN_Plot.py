# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(123)

df = pd.read_csv("cleandata.csv")
# Holds the position of each player in the dataset by index
targets = df.pos

# Import the PCA dataset
components = pd.read_csv("df_pca.csv")
# Take only the first three dimensions given by the PCA
components = components.iloc[:,0:3]

from sklearn import neighbors
from sklearn.model_selection import train_test_split  

# Convert from a pandas dataframe to a numpy array
X = components.values
y = targets.values

# Split the data into 80% as training data and 20% as testing data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  

# Searches for the nearest 8 points and classifies based which position
# represents the majority of those 8 nearest points
clf = neighbors.KNeighborsClassifier(n_neighbors = 8)
test = clf.fit(X_train, y_train)

# Shows what the model predicted for each test case
y_pred = clf.predict(X_test)

# Allows one to see what position a player was in the plot.ly graph by hovering over the points
players = ((y_test).tolist())

import plotly.plotly as py
import plotly.graph_objs as go

# change the color of the test dataset based on what the KNN model predicted 
color=np.array(['rgb(000,000,000)']*X_test.shape[0]) #black

for i in range(len(y_pred)):
    if y_pred[i] == 'SG':
        color[i] = 'rgb(255,0,0)' #red
    elif y_pred[i] == 'SF':
        color[i]= 'rgb(0,192,0)' #green
    elif y_pred[i] == 'PF':
        color[i]='rgb(255,224,32)' #yellow
    elif y_pred[i] == 'C':
        color[i]='rgb(0,0,255)' #blue

# plot only the test dataset
x = X_test[:,0]
y = X_test[:,1]
z = X_test[:,2]
import numpy as np
py.sign_in('nthuy', 'upwufOB7qO0LmdxqEWo6')

trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    text=players,
    mode='markers',
    marker=dict(
        size=12,
        color=color.tolist(),                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

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
py.iplot(fig, filename='3d-scatter-colorscale')