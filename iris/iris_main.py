import pandas as pd
from random import shuffle

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
print(dataset.shape)
arr = dataset.values
shuffle(arr)
tr, ts = arr[:120,:], arr[120:,:]
tr_pd = pd.DataFrame(data=tr,columns=names)
ts_pd = pd.DataFrame(data=ts,columns=names)

ts_pd.to_csv("test_iris.csv", sep='\t')
tr_pd.to_csv("train_iris.csv", sep='\t')
