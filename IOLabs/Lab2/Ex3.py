import math

from pandas import read_csv
import matplotlib.pyplot as plt

iris_dataset = read_csv("iris.csv")

iris_sepal_dataset = iris_dataset.drop(columns=["petal.length", "petal.width"])

# Making an original dataset plot
iris1x = []
iris1y = []
for index in range(0, iris_sepal_dataset.index.size):
    if iris_sepal_dataset.values[index, iris_sepal_dataset.columns.size - 1] == "Setosa":
        iris1x.append(iris_sepal_dataset.values[index, 0])
        iris1y.append(iris_sepal_dataset.values[index, 1])

iris2x = []
iris2y = []
for index in range(0, iris_sepal_dataset.index.size):
    if iris_sepal_dataset.values[index, iris_sepal_dataset.columns.size - 1] == "Versicolor":
        iris2x.append(iris_sepal_dataset.values[index, 0])
        iris2y.append(iris_sepal_dataset.values[index, 1])

iris3x = []
iris3y = []
for index in range(0, iris_sepal_dataset.index.size):
    if iris_sepal_dataset.values[index, iris_sepal_dataset.columns.size - 1] == "Virginica":
        iris3x.append(iris_sepal_dataset.values[index, 0])
        iris3y.append(iris_sepal_dataset.values[index, 1])

plt.scatter(iris1x, iris1y)
plt.scatter(iris2x, iris2y)
plt.scatter(iris3x, iris3y)

plt.title("Original dataset")
plt.legend(["Setosa", "Versicolor", "Virginica"])
plt.xlabel("Sepal height (cm)")
plt.ylabel("Sepal width (cm)")
plt.show()

# Making a min-max dataset plot
x_min = 10
for index in range(iris_sepal_dataset.index.size):
    if iris_sepal_dataset.values[index, 0] < x_min:
        x_min = iris_sepal_dataset.values[index, 0]
x_max = 0
for index in range(iris_sepal_dataset.index.size):
    if iris_sepal_dataset.values[index, 0] > x_max:
        x_max = iris_sepal_dataset.values[index, 0]
y_min = 10
for index in range(iris_sepal_dataset.index.size):
    if iris_sepal_dataset.values[index, 1] < y_min:
        y_min = iris_sepal_dataset.values[index, 1]
y_max = 0
for index in range(iris_sepal_dataset.index.size):
    if iris_sepal_dataset.values[index, 1] > y_max:
        y_max = iris_sepal_dataset.values[index, 1]

iris1x_normalized = []
iris1y_normalized = []
for index in range(0, len(iris1x) - 1):
    iris1x_normalized.append(round((iris1x[index] - x_min) / (x_max - x_min), 2))
for index in range(0, len(iris1y) - 1):
    iris1y_normalized.append(round((iris1y[index] - y_min) / (y_max - y_min), 2))

iris2x_normalized = []
iris2y_normalized = []
for index in range(0, len(iris2x) - 1):
    iris2x_normalized.append(round((iris2x[index] - x_min) / (x_max - x_min), 2))
for index in range(0, len(iris2y) - 1):
    iris2y_normalized.append(round((iris2y[index] - y_min) / (y_max - y_min), 2))

iris3x_normalized = []
iris3y_normalized = []
for index in range(0, len(iris3x) - 1):
    iris3x_normalized.append(round((iris3x[index] - x_min) / (x_max - x_min), 2))
for index in range(0, len(iris3y) - 1):
    iris3y_normalized.append(round((iris3y[index] - y_min) / (y_max - y_min), 2))

plt.scatter(iris1x_normalized, iris1y_normalized)
plt.scatter(iris2x_normalized, iris2y_normalized)
plt.scatter(iris3x_normalized, iris3y_normalized)

plt.title("Min-max normalized dataset")
plt.legend(["Setosa", "Versicolor", "Virginica"])
plt.xlabel("Sepal height (cm)")
plt.ylabel("Sepal width (cm)")
plt.show()

# Making a z-score scaled dataset plot
x_mean = round(sum(iris_sepal_dataset.values[:, 0]) / iris_sepal_dataset.index.size, 2)
y_mean = round(sum(iris_sepal_dataset.values[:, 1]) / iris_sepal_dataset.index.size, 2)

x_variance = 0
for index in range(0, iris_sepal_dataset.index.size):
    x_variance += (float(iris_sepal_dataset.values[index, 0]) - x_mean) ** 2
x_variance = x_variance / iris_sepal_dataset.index.size

y_variance = 0
for index in range(0, iris_sepal_dataset.index.size):
    y_variance += (float(iris_sepal_dataset.values[index, 1]) - x_mean) ** 2
y_variance = y_variance / iris_sepal_dataset.index.size

x_standard = math.sqrt(x_variance)
y_standard = math.sqrt(y_variance)

iris1x_zscore = []
iris1y_zscore = []
for index in range(0, len(iris1x) - 1):
    iris1x_zscore.append(round((iris1x[index] - x_mean) / x_standard, 2))
for index in range(0, len(iris1y) - 1):
    iris1y_zscore.append(round((iris1y[index] - y_mean) / y_standard, 2))

iris2x_zscore = []
iris2y_zscore = []
for index in range(0, len(iris2x) - 1):
    iris2x_zscore.append(round((iris2x[index] - x_mean) / x_standard, 2))
for index in range(0, len(iris2y) - 1):
    iris2y_zscore.append(round((iris2y[index] - y_mean) / y_standard, 2))

iris3x_zscore = []
iris3y_zscore = []
for index in range(0, len(iris3x) - 1):
    iris3x_zscore.append(round((iris3x[index] - x_mean) / x_standard, 2))
for index in range(0, len(iris3y) - 1):
    iris3y_zscore.append(round((iris3y[index] - y_mean) / y_standard, 2))

plt.scatter(iris1x_zscore, iris1y_zscore)
plt.scatter(iris2x_zscore, iris2y_zscore)
plt.scatter(iris3x_zscore, iris3y_zscore)

plt.title("Z-score scaled dataset")
plt.legend(["Setosa", "Versicolor", "Virginica"])
plt.xlabel("Sepal height (cm)")
plt.ylabel("Sepal width (cm)")
plt.show()
