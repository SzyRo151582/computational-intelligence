import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

iris_data_frame = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(iris_data_frame.values, train_size=0.7, random_state=7565)

print(f"Iris training set: \n{train_set}")
print(f"\nIris test set: \n{test_set}")

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

dtc = tree.DecisionTreeClassifier()
dtc = dtc.fit(X=train_inputs, y=train_classes)

ai_score = dtc.score(test_inputs, test_classes)
print(f"The accuracy of the decision tree classifier: {ai_score}")

tree.plot_tree(dtc)
plt.show()

predictions = dtc.predict(test_inputs)
c_matrix = confusion_matrix(y_true=test_classes, y_pred=predictions)
print(c_matrix)
