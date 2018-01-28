from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import pandas as pd
import numpy as np
from sklearn import metrics

dataset = pd.read_csv('test_iris.csv', sep=r'\t')
target = pd.read_csv('train_iris.csv', sep=r'\t')
#print(dataset.groupby('class').size())

"""
box and whiskers plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

histograms
dataset.hist()

scatter_matrix(dataset)
plt.show()
"""

lb = { 'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2 }
x_lb = dataset['class'].map(lb).values
y_lb = target['class'].map(lb).values

del dataset['class']
x_val = dataset.values

del target['class']
y_val = target.values

model = LogisticRegression()
model.fit(x_val, x_lb)
print(model)

predict = model.predict(x_val)
print(predict)

print(metrics.classification_report(x_lb, predict))
print(metrics.confusion_matrix(x_lb, predict))
print(metrics.accuracy_score(x_lb, predict))
