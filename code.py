import numpy as np
import pickle

import time
import multiprocessing

#from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, f1_score, recall_score
import matplotlib.pyplot as plt

cores = multiprocessing.cpu_count()
print("%d number of cores are available"%(cores))

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

# rs: F1 Scores nm: Model names
rs = []
nm = []

"""
    fname: Model Type
    name: Model Type Abbreviations
    model: Model Type Function
"""
def model_pr(fname, name, model):
    # t1: Start time t2: Stop time
    t1 = time.time()
    model.fit(x_tr, x_lb)
    y_pred = model.predict(y_ts)
    res = f1_score(y_lb, y_pred, average = 'micro')
    rs.append(res*100)
    nm.append(name)
    print(("Precision is %.3f")%precision_score(y_lb, y_pred, average = 'micro'))
    print(("Recall Score is %.3f")%recall_score(y_lb, y_pred, average = 'micro'))
    print(fname, "%.3f"%res)
    t2 = time.time()
    print("Run time is %.4f \n"% (t2-t1))

print("Let's start computations")
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
