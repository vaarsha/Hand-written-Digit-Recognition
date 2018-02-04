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
import matplotlib.pyplot as plt

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

def model_pr(fname, name, model):
    model.fit(x_tr, x_lb)
    y_pred = model.predict(y_ts)
    res = accuracy_score(y_lb, y_pred)
    rs.append(res)
    nm.append(name)
    print(confusion_matrix(y_ts, y_pred))
    print(classification_report(y_ts, y_pred))
    print(fname, "%.3f"%res)

model_pr("Logistic Regression", "LR", LogisticRegression())
model_pr("Naive Bayes", "NB", GaussianNB())
model_pr("Classification and Regression Trees", "CR", DecisionTreeClassifier())
model_pr("Support Vector Machines", "SVM", SVC())
model_pr("K-Nearest Neighbors", "KNN", KNeighborsClassifier())
model_pr("Linear Discriminant Analysis", "LDA", LinearDiscriminantAnalysis())

# Plotting graph
y_axis = np.arange(len(nm))

plt.bar(y_axis, rs, align='center')
plt.xticks(y_axis, nm)
plt.ylabel("Accuracy Score")
plt.title("Model Comparisons")

plt.show()
