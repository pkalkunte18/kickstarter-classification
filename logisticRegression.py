# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:11:04 2022

@author: saipr
"""

import pandas as pd 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

#--------------------DATA LOADING / CLEANING--------------------------------#
#load in the data - 1 year of kickstarter data, 2021
#data = pd.read_csv("C:\\Users\\saipr\\Desktop\\Programming\\Kickstarter Classification\\cleaned.csv", index_col = 0) #for vee's laptop
#for running after downloading everything on the github repository - load in each "cleaned"
data = pd.read_csv("cleaned1.csv")
data2 = pd.read_csv("cleaned2.csv")
data3 = pd.read_csv("cleaned3.csv")
data = data.concat(data2, ignore_index = True)
data = data.concat(data3, ignore_index = True)
#now we have the full merged file
print(data.head())
print(data.columns)
print(data.state.unique())

#turn state into a binary variable
state2 = [0 for i in range(0, len(data))] #a column of zeroes
for s in range(0, len(data.state)):
    #print(data.state[s])
    if(data.state[s] == "successful"):
        state2[s] = 1
data['state2'] = state2 #state 2: electric boogaloo; basically a logistic regression is binary, so we're not making distinctions here
print(data.info())

#and enocode those categories 
for col in data.columns: #for each column
    #check if the datatype is numerical - if so, skip it
    if(data[col].dtype == 'int64'):
        continue 
    #make a new label encoder
    le = preprocessing.LabelEncoder()
    cats = data[col].unique() #categorial values
    le.fit(cats) #fit the encoader to this
    newCols = col + "_encoded" #new column name
    data[newCols] = le.transform(data[col].values).tolist() #add the encoded column
print(data.info()) #results of our labor - success!

#our x (features) and y (label) data
y = data.state2
#drop our label + all the categorial data (so x is just the encoded categories + numerical data)
X = data.drop(['state', 'country', 'subcats', 'is_starrable', 'spotlight', 'staff_pick', 'state_encoded', 'state2'], axis = 1)

#divide into test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size=0.3)

print(X_train.columns)

#--------------------MODEL: Logistic Regression----------------------------#
#tutorial: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
model = LogisticRegression(random_state=0).fit(X_train, y_train)
#make your predicted values
y_hat = model.predict(X_test)
#and scoring functions
accuracy = accuracy_score(y_test, y_hat)*100
cm = confusion_matrix(y_test, y_hat)

#view our metrics
print("Confusion matrix:\n", cm)
cm_display = ConfusionMatrixDisplay(cm).plot()

#--------------------SCORING METRICS--------------------------------#
#and now dive a little deeper into the scoring:
#tutorial: https://blog.nillsf.com/index.php/2020/05/23/confusion-matrix-accuracy-recall-precision-false-positive-rate-and-f-scores-explained/
trueNeg = cm[0][0] #1 is successful, 0 is else
truePos = cm[1][1] #bottom right
falsePos = cm[0][1] #top right
falseNeg = cm[1][0] #bottom left

accuracy = (truePos + trueNeg) / len(y_test)
recall = truePos / (truePos + falseNeg)
precision = truePos / (truePos + falsePos)
falsePosRate = falsePos / (trueNeg + falsePos)
f1 = (precision*recall) / (precision + recall)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("False Positive Rate:", falsePosRate)
print("F1 Rate (Harmonic Mean):", f1)
