# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import random
random.seed(123)

df = pd.read_csv("cleandata.csv")

# Principal Component Analysis with all components kept, used on all columns
# kept that may have an influence on determining a player's position
pca = PCA()

df_pca = pca.fit(df.iloc[:,4:37]).transform(df.iloc[:,4:37])

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

# Convert from numpy.ndarray to pandas dataframe to make a csv file
df_pca = (pd.DataFrame(df_pca))

df_pca.to_csv('df_pca.csv', index=False)

# Used to allow for all indexes of the multi-dimensional matrix to be 
# printed to the screen
#np.set_printoptions(threshold=np.inf)
#print abs( pca.components_ )

import matplotlib.pyplot as plt 


pcavar = (pca.explained_variance_ratio_)
y_pos = np.arange(len(pcavar))
cumul = []
for i in range(len(pcavar)):
    if i != 0:
        cumul.append(cumul[i-1] + pcavar[i])
    else:
        cumul.append(pcavar[i])
print cumul

lines, = plt.plot(y_pos, pcavar, color='b', linewidth='1.5', label='Individual PCA Ratio')
bars = plt.bar(y_pos, cumul, color='y', edgecolor='black', linewidth='1.1', label='Cumulative PCA Ratio')

plt.axis([-1, 33, 0, 1.1])
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Most Important PCAs')

# Create a legend for the first line.
first_legend = plt.legend(handles=[lines], loc=1)
# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)
plt.legend(handles=[lines], loc=2)
# Create another legend for the bar graph.
plt.legend(handles=[bars], loc=4)

ax = plt.axes()
ax.yaxis.grid(True, color='gray', linestyle='dashed')
ax.set_axisbelow(True)

plt.show()

