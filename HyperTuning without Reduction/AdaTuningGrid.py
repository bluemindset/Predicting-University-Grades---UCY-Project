# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 19:55:43 2019

@author: Admin
"""


from SelectedData import *  
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import *
from sklearn.tree import DecisionTreeClassifier


MIN_EST = 498
MAX_EST = 500
TUNE_P = [{'n_estimators': range(MIN_EST,MAX_EST)}]
TUNE_P2 = [{'n_estimators': [270,285,500]}]
clf = GridSearchCV(AdaBoostClassifier(), TUNE_P2,cv= 2,scoring = 'recall')   


print("# Tuning hyper-parameters for n_estimator %s TO %s" %(MIN_EST,    MAX_EST) )
print()
clf = clf.fit(xtrain, ytrain)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.6f (+/-%0.6f) for %r" % (mean, std * 2, params))
    print()
print("Grid scores on development set:") 
print(clf.best_params_)  
ynew =  clf.predict(xtest)
accuracy = accuracy_score(ynew, ytest)
print ("Accuracy is %s", accuracy)
print(classification_report(ynew, ytest,digits = 6))



