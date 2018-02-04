import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection

#dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#data = pd.read_csv(dataset_url, names=names)

data = pd.read_csv('test.csv', sep=r'\t')

lb = { 'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2 }
y = data['class'].map(lb).values

x = data.drop('class', axis=1)
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.2,
                                          random_state=123, stratify=y)
rs = []
nm = []

def model_pr(name, model):
    model.fit(x_tr, y_tr)
    y_pred = model.predict(x_ts)
    res = accuracy_score(y_ts, y_pred)
    rs.append(res)
    nm.append(name)
    print(name, "%.3f"%res)

model_pr("Logistic Regression", LogisticRegression())
model_pr("Naive Bayes", GaussianNB())
model_pr("Classification and Regression Trees", DecisionTreeClassifier())
model_pr("Support Vector Machines", SVC())
model_pr("K-Nearest Neighbors", KNeighborsClassifier())
model_pr("Linear Discriminant Analysis", LinearDiscriminantAnalysis())
#cm = confusion_matrix(y_ts, y_pred)

"""
kfold = model_selection.KFold(n_splits=10, random_state=7)
rs = cross_val_score(model, x_tr, y_tr, cv = kfold, scoring =
        'accuracy')
print(rs)

scaler = preprocessing.StandardScaler().fit(x_tr)
x_tr_scale = scaler.transform(x_tr)
x_ts_scale = scaler.transform(x_ts)
"""
