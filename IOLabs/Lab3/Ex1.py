import pandas as pd
from sklearn.model_selection import train_test_split

iris_data_frame = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(iris_data_frame.values, train_size=0.7, random_state=609299)

def classify_iris(sepal_length, sepal_width, petal_length, petal_width):
    if petal_width < 0.9:
        return "Setosa"
    elif petal_length >= 4.9:
        return "Virginica"
    else:
        return "Versicolor"

good_predictions = 0
quantity_of_tests = test_set.shape[0]

for i in range(0, quantity_of_tests):
    if classify_iris(sepal_length=test_set[i, 0], sepal_width=test_set[i, 1], petal_length=test_set[i, 2],
                     petal_width=test_set[i, 3]) == test_set[i, 4]:
        good_predictions += 1

print(good_predictions)
print(good_predictions / quantity_of_tests * 100, '%')
