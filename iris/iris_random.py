import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv(dataset_url, names=names)

y = data['class']
x = data.drop('class', axis=1)
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.2,
                                          random_state=123, stratify=y)


classifier = LogisticRegression()
classifier.fit(x_tr, y_tr)
y_pred = classifier.predict(x_ts)
confusion_matrix = confusion_matrix(y_ts, y_pred)
print(confusion_matrix)


scaler = preprocessing.StandardScaler().fit(x_tr)
x_tr_scale = scaler.transform(x_tr)
x_ts_scale = scaler.transform(x_ts)


