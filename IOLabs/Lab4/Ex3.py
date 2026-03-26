import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

diabetes_dataset = pd.read_csv("diabetes.csv")

train_set, test_set = train_test_split(diabetes_dataset.values, train_size=0.7, random_state=1)
train_inputs = train_set[:, 0:8]
train_classes = train_set[:, 8]
test_inputs = test_set[:, 0:8]
test_classes = test_set[:, 8]

mlp = MLPClassifier(hidden_layer_sizes=(6, 3), activation="relu", max_iter=500)
mlp.fit(train_inputs, train_classes)

train_predict = mlp.predict(train_inputs)
test_predict = mlp.predict(test_inputs)
mlp_score = mlp.score(test_inputs, test_classes)

print(f"Accuracy of Multilayer perceptron classifier: {round(mlp_score * 100, 2)}%.")
print(confusion_matrix(test_classes,test_predict))

# Z reguły ten klasyfikator jest dokładniejszy od pozostałych z poprzedniego zadania.
# FN będzie gorszy, pacjent bedzie nadal robił to co robi. Z tego powodu jego/jej stan może sie pogarszać.

mlp2 = MLPClassifier(hidden_layer_sizes=(7, 5), activation="relu", max_iter=500)
mlp2.fit(train_inputs, train_classes)
train_predict2 = mlp2.predict(train_inputs)
test_predict2 = mlp2.predict(test_inputs)
mlp_score2 = mlp2.score(test_inputs, test_classes)

print(f"Accuracy of Multilayer perceptron classifier: {round(mlp_score2 * 100, 2)}%.")

mlp3 = MLPClassifier(hidden_layer_sizes=(7, 5), activation="logistic", max_iter=500)
mlp3.fit(train_inputs, train_classes)
train_predict3 = mlp3.predict(train_inputs)
test_predict3 = mlp3.predict(test_inputs)
mlp_score3 = mlp3.score(test_inputs, test_classes)

print(f"Accuracy of Multilayer perceptron classifier: {round(mlp_score3 * 100, 2)}%.")
