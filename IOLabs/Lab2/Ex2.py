from pandas import read_csv
from sklearn.preprocessing import StandardScaler

iris_dataset = read_csv("iris.csv")

columns = iris_dataset.columns.values

X = iris_dataset.loc[:, columns].values
Y = iris_dataset.loc[:, ['variety']].values

X = StandardScaler().fit_transform(X)

def var(list):
  n = len(list)
  average = sum(list)/n
  return sum((list[i] - average)**2 for i in range(1,n))/n
