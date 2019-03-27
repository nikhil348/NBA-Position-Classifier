# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(123)

df = pd.read_csv("cleandata.csv")
# Holds the position of each player in the dataset by index
targets = df.pos

# Take only the first three dimensions given by the PCA
components = pd.read_csv("df_pca.csv")
components = components.iloc[:,0:3]

from sklearn import neighbors
from sklearn.model_selection import train_test_split  

# Convert from a pandas dataframe to a numpy array
X = components.values
y = targets.values

# Split the data into 80% as training data and 20% as testing data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  

# Searches for the nearest 5 points and classifies based which position
# represents the majority of those 5 nearest points
clf = neighbors.KNeighborsClassifier(n_neighbors = 26)
test = clf.fit(X_train, y_train)
print test
# Shows what the model predicted for each test case
y_pred = clf.predict(X_test)
print y_pred
print X_test
# Confusion matrix shows how many of each position prediction was
# made for each actual NBA position. i.e. this outputs a 5x5 matrix
# where the first row represents how many centers were correctly predicted
# by the machine, along with how many centers were incorrectly placed
# at the 4 other positions.
# The classification report conveys this in decimals
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


error = []

# Calculating error for K values between 1 and 40
# Helps to figure out which K values yield the smallest error
for i in range(1, 40):  
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  