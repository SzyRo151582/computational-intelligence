import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

iris_dataset = pd.read_csv("iris.csv")

train_set, test_set = train_test_split(iris_dataset.values, test_size=0.7, random_state=100)
train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

clf1 = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
clf1.fit(train_inputs, train_classes)

train_predictions1 = clf1.predict(train_inputs)
test_predictions1 = clf1.predict(test_inputs)
train_score1 = accuracy_score(train_predictions1, train_classes)
test_score1 = accuracy_score(test_predictions1, test_classes)
print(f"Score on test data using first Multilayer perceptron: {round(test_score1 * 100, 2)}%.")

clf2 = MLPClassifier(hidden_layer_sizes=(3,), max_iter=1000)
clf2.fit(train_inputs, train_classes)

train_predictions2 = clf2.predict(train_inputs)
test_predictions2 = clf2.predict(test_inputs)
train_score2 = accuracy_score(train_predictions2, train_classes)
test_score2 = accuracy_score(test_predictions2, test_classes)
print(f"Score on test data using second Multilayer perceptron: {round(test_score2 * 100, 2)}%.")

clf3 = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=1000)
clf3.fit(train_inputs, train_classes)

train_predictions3 = clf3.predict(train_inputs)
test_predictions3 = clf3.predict(test_inputs)
train_score3 = accuracy_score(train_predictions3, train_classes)
test_score3 = accuracy_score(test_predictions3, test_classes)
print(f"Score on test data using third Multilayer perceptron: {round(test_score3 * 100, 2)}%.")

if test_score1 > test_score2 and test_score1 > test_score3:
    print("First Multilayer perceptron has best score.")
elif test_score2 > test_score1 and test_score2 > test_score3:
    print("Second Multilayer perceptron has best score.")
else:
    print("Third Multilayer perceptron has best score.")
    