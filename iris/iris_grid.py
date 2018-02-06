import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
#from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn import metrics


data = pd.read_csv('test.csv', sep=r'\t')

lb = { 'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2 }
y = data['class'].map(lb).values

x = data.drop('class', axis=1)
"""
    x_tr: Training feature values
    x_ts: Testing feature values
    y_tr: Training label
    y_ts: Testing label
"""
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.2,
                                          random_state=0)
rs = []
nm = []

parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

clf = GridSearchCV(SVC(decision_function_shape='ovr'), parameters, cv=5)
clf.fit(x_tr, y_tr)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
best_score = -999
best_params = {}

for params, mean, score in clf.grid_scores_:
    if mean > best_score:
        best_score = mean
        best_params = params

    print("%0.3f (+/-%0.03f) for %r"
          % (mean, score.std() * 2, params))

print(("Best score is %0.3f")%best_score, " for parameters: ",best_params)

y_pred = clf.predict(x_ts)
print(classification_report(y_ts, y_pred))
