# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:41:24 2019

@author: Admin
"""

from sklearn.ensemble import GradientBoostingClassifier
from SelectedData import *  
from sklearn.metrics import *
'''
max = 0
estimators = 0
for x in range(10,1000):
    
    clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=x, subsample=1.0, criterion='friedman_mse',
                                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None,
                                 verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
    print("# Tuning hyper-parameters for n_estimator %s" % x)
    print()
    clf = clf.fit(xtrain, ytrain)
    print("Grid scores on development set:") 
    print()  
    ynew =  clf.predict(xtest)
    accuracy = accuracy_score(ynew, ytest)
    print(accuracy)
    print(estimators)
    
    if (max<accuracy):
            max = accuracy_score(ynew, ytest)
            estimators = x

print("Estimators are %s" % estimators)
print("Accuracy Score is %s" % max)

#best estimator is 500 with 81,4%
'''
max = 0
estimators = 0
clf = GradientBoostingClassifier(loss='exponential', learning_rate=0.005, n_estimators=1000, subsample=1.0, criterion='friedman_mse',
                                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=32, min_impurity_decrease=0.0, min_impurity_split=None, 
                                 init=None, random_state=None, max_features=None,
                                 verbose=0, max_leaf_nodes=100, warm_start=False, presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
print("# Tuning hyper-parameters for n_estimator %s" % 285)
print()
clf = clf.fit(xtrain, ytrain)
print("Grid scores on development set:") 
print()  
ynew =  clf.predict(xtest)
accuracy = accuracy_score(ynew, ytest)
print(accuracy)

print("Estimators are %s" % estimators)
print("Accuracy Score is %s" % max)
#best estimator is with dTree classifier of max depth 80,099%




