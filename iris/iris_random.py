import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
import matplotlib.pyplot as plt

import time
import multiprocessing
from sklearn import metrics

#dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#data = pd.read_csv(dataset_url, names=names)

cores = multiprocessing.cpu_count()

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
                                          random_state=123, stratify=y)
rs = []
nm = []

# t1: Start time t2: Stop time

def model_pr(fname, name, model):
    t1 = time.time()
    model.fit(x_tr, y_tr)
    y_pred = model.predict(x_ts)
    res = metrics.f1_score(y_ts, y_pred, average = 'micro')
    rs.append(res*100)
    nm.append(name)
    print(fname, "%.3f"%res)
    print(("Precision is %.3f")%metrics.precision_score(y_ts, y_pred, average = 'micro'))
    print(("Recall Score is %.3f")%metrics.recall_score(y_ts, y_pred, average = 'micro'))
    t2 = time.time()
    print("Run time is %.4f \n"% (t2-t1))

model_pr("Logistic Regression", "LR", LogisticRegression(n_jobs = cores))
model_pr("Naive Bayes", "NB", GaussianNB())
model_pr("Classification and Regression Trees", "CR", DecisionTreeClassifier())
model_pr("Support Vector Machines", "SVM", SVC())
model_pr("K-Nearest Neighbors", "KNN", KNeighborsClassifier(n_jobs = cores))
model_pr("Linear Discriminant Analysis", "LDA", LinearDiscriminantAnalysis())
model_pr("Random Forest", "RF", RandomForestClassifier(n_estimators=10, n_jobs = cores))

# Plotting graph

y_axis = np.arange(len(nm))

plt.bar(y_axis, rs, align='center')
plt.xticks(y_axis, nm)
plt.ylabel("F1 Score")
plt.title("Model Comparisons")

plt.show()
