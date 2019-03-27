#!/usr/bin/env python
# coding: utf-8

# In[114]:


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import pandas as pd
import random

random.seed(123)

df = pd.read_csv("cleandata.csv")
targets = df.pos



X = df.iloc[:,4:37]
y = targets


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  


# In[115]:


df


# In[116]:


random.seed(123)

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print"Accuracy:",metrics.accuracy_score(y_test, y_pred)


# In[117]:


#import sys
#!{sys.executable} -m pip install graphviz


# In[118]:


#import sys
#!{sys.executable} -m pip install pydotplus


# In[119]:


##all of this code was to visualize the decision tree; since it is not being used for final analysis, it is 
##commented out since this code takes very long to run

#from IPython.display import SVG
#from sklearn import tree
#from graphviz import Source
#from IPython.display import display
#import pydotplus

#data_feature_names = list(X)

#dot_data = tree.export_graphviz(clf,
#                                feature_names=data_feature_names,
#                               out_file=None,
#                                filled=True,
#                                rounded=True)


#graph = Source(tree.export_graphviz(clf, out_file=None
#   , feature_names=np.asarray(list(X)), class_names=['0', '1', '2'] 
#   , filled = True))
#graph = pydotplus.graph_from_dot_data(dot_data)
#display(graph)


# In[120]:


#graph.write_png('tree.png')


# In[121]:


random.seed(123)

# Create Decision Tree classifer object
clf2 = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf2 = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[122]:


#dot_data = tree.export_graphviz(clf2,
#                                feature_names=data_feature_names,
#                                out_file=None,
#                                filled=True,
#                                rounded=True)


#graph = Source(tree.export_graphviz(clf, out_file=None
#   , feature_names=np.asarray(list(X)), class_names=['0', '1', '2'] 
#   , filled = True))
#graph = pydotplus.graph_from_dot_data(dot_data)
#display(graph)
#graph.write_png('tree2.png')


# In[123]:


random.seed(123)

from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[124]:


from pprint import pprint
pprint(clf.get_params())


# In[125]:


import matplotlib.pyplot as plt

random.seed(123)

#feature_imp = pd.Series(clf.feature_importances_,index=list(df.iloc[:,4:37].columns.values)).sort_values(ascending=False)
feature_imp = pd.Series(clf.feature_importances_,index=list(df.iloc[:,4:37].columns.values))
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[126]:


random.seed(123)

feature_imp2 = pd.Series(clf.feature_importances_,index=list(df.iloc[:,4:37].columns.values)).sort_values(ascending=False)
#feature_imp = pd.Series(clf.feature_importances_,index=list(df.iloc[:,4:37].columns.values))
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
# Creating a bar plot
sns.barplot(x=feature_imp2, y=feature_imp2.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[127]:


feature_imp


# In[128]:


random.seed(123)

import numpy as np

indices = np.where(feature_imp > 0.02)
indices
list(indices)


# In[129]:


random.seed(123)

newX = X.iloc[:,[ 0,  1,  3,  4, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 31, 32]]


# In[130]:


random.seed(123)

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(newX, y, test_size=0.20)  


# In[131]:


random.seed(123)

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=300)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train_new,y_train_new)

y_pred=clf.predict(X_test_new)

print("Accuracy:",metrics.accuracy_score(y_test_new, y_pred))


# In[132]:


random.seed(123)

feature_imp = pd.Series(clf.feature_importances_,index=list(newX.columns.values)).sort_values(ascending=False)
#feature_imp = pd.Series(clf.feature_importances_,index=list(df.iloc[:,4:37].columns.values))
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[133]:


random.seed(123)

from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier

'''
this code was to create the graph that shows error rate vs number of trees. i have commented it out because it takes
very long to run, since it does many many random forest iterations

ensemble_clfs = [ ("RandomForestClassifier, max_features=None",RandomForestClassifier(warm_start=True, max_features=None, oob_score=True,))]


error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)


min_estimators = 100
max_estimators = 500

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X_train_new,y_train_new)
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))


for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()
'''


# In[134]:


import time
random.seed(123)
start_time = time.time()

clffinal=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clffinal.fit(X_train_new,y_train_new)

y_pred=clffinal.predict(X_test_new)

print "Accuracy:",metrics.accuracy_score(y_test_new, y_pred)
print("%s seconds" % (time.time() - start_time))


# In[135]:


random.seed(123)

start_time = time.time()

clf=RandomForestClassifier(n_estimators=300)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train_new,y_train_new)

y_pred=clf.predict(X_test_new)

print("Accuracy:",metrics.accuracy_score(y_test_new, y_pred))
print("%s seconds" % (time.time() - start_time))


# In[136]:


random.seed(123)

start_time = time.time()

clf=RandomForestClassifier(n_estimators=600)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train_new,y_train_new)

y_pred=clf.predict(X_test_new)

print("Accuracy:",metrics.accuracy_score(y_test_new, y_pred))
print("%s seconds" % (time.time() - start_time))


# In[137]:


len(list(newX))


# In[138]:


len(list(X))


# In[139]:


newX.shape


# In[140]:


df.shape


# In[142]:


import time
random.seed(123)
start_time = time.time()

clffinal=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clffinal.fit(X_train_new,y_train_new)

y_pred=clffinal.predict(X_test_new)

print "Accuracy:",metrics.accuracy_score(y_test_new, y_pred)
print("%s seconds" % (time.time() - start_time))


# In[148]:


from sklearn.metrics import confusion_matrix

confuse = confusion_matrix(y_test_new, y_pred)
confuse


# In[151]:


confusedf = pd.DataFrame(data=confuse[0:,0:])
confusedf.columns = ['Predicted PG','Predicted SG','Predicted SF',
                     'Predicted PF','Predicted C']
confusedf.rename(index={0:'Actual PG',1:'Actual SG', 2:'Actual SF', 3:'Actual PF', 4:'Actual C'}, inplace=True)
confusedf


# In[152]:


df


# In[175]:


grouped = df.groupby('pos')
dpercent = grouped['drbpercent'].agg([np.mean, np.std])
dpercentdf = pd.DataFrame(dpercent)
dpercentdf.columns = ['Mean Defensive Rebound %', 'Standard Deviation']
dpercentdf.sort_values(by=['Mean Defensive Rebound %'], ascending = False)


# In[176]:


astpercent = grouped['astpercent'].agg([np.mean, np.std])
astpercentdf = pd.DataFrame(astpercent)
astpercentdf.columns = ['Mean Assist %', 'Standard Deviation']
astpercentdf.sort_values(by=['Mean Assist %'], ascending = False)


# In[177]:


y_pred


# In[ ]:




