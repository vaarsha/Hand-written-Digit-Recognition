import numpy as np
import pickle
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

fl = open('dataset.pkl', 'rb')
dataset = pickle.load(fl)
fl.close()
print("loading dataset complete")

x_tr = dataset['tr_img']
x_lb = dataset['tr_label']
y_ts = dataset['ts_img']
y_lb = dataset['ts_label']

x_tr = x_tr / 255
y_ts = y_ts /255

rs = []
nm = []

def model_pr(name, model):
    model.fit(x_tr, x_lb)
    y_pred = model.predict(y_ts)
    res = accuracy_score(y_lb, y_pred)
    rs.append(res)
    nm.append(name)
    print(name, "%.3f"%res)

model_pr("Logistic Regression", LogisticRegression())
model_pr("Naive Bayes", GaussianNB())
model_pr("Classification and Regression Trees", DecisionTreeClassifier())
model_pr("Support Vector Machines", SVC())
model_pr("K-Nearest Neighbors", KNeighborsClassifier())
model_pr("Linear Discriminant Analysis", LinearDiscriminantAnalysis())
