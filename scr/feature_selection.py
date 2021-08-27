# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 21:40:45 2021

@author: Smegn
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy import sparse 
#model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

#feature selection
from sklearn.inspection import permutation_importance


plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})


# Load the dataset
dataset = pd.read_csv(r'C:/Users/Smegn/Documents/GitHub/Breast-Cancer/data/data.csv')
X = dataset.iloc[:, 2:32].values
y = dataset.iloc[:, 1].values

#print(X)
#print(y)

# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset into the Training set and Test set using 80% and 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Visualize
sns.pairplot(dataset.iloc[:,1:5], hue='diagnosis')
   #save fig in folder
#plt.savefig("C:/Users/Smegn/Documents/GitHub/Breast-Cancer/image/fig_1")

#creat function to model
def model(X_train, y_train):
    #logestic regression
    log=LogisticRegression(random_state = 0)
    log.fit(X_train, y_train)
    
    # DecisionTreeClassifier
    DTC = DecisionTreeClassifier(random_state=0)
    DTC.fit(X_train, y_train)
    
     # Random forest Classifier
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)
    #print the model accuracy on the training data
    print('[0] Logestic Regression Training Accuracy:', log.score(X_train, y_train))
    print('[1] Decision Tree Classifier Training Accuracy:', DTC.score(X_train, y_train))
    print('[2] Random forest Classifier Training Accuracy:', forest.score(X_train, y_train))
    return log, DTC,forest
 # model accuracy
model(X_train, y_train)
 # Random forest Classifier
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)

#feature importance using Mean decrease in impurity
# selection using Random forest
importances = forest.feature_importances_
    #(importances)
feature_names = [f'feature {i}' for i in range(X.shape[1])]
forest_importances = pd.Series(importances, index=feature_names)
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
start_time=time.time()
elapsed_time = time.time() - start_time
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
#plt.savefig("C:/Users/Smegn/Documents/GitHub/Breast-Cancer/image/fig_2")

######
#from sklearn.inspection import permutation_importance

start_time = time.time()
result = permutation_importance(forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: "
      f"{elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
##plt.savefig("C:/Users/Smegn/Documents/GitHub/Breast-Cancer/image/fig_3")
print(feature_names)
####
#logestic regression
#log=LogisticRegression(random_state = 0)
#log.fit(X_train, y_train)
#plt.plot(log)