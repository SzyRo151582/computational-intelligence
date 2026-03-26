import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


iris_data_frame = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(iris_data_frame.values, train_size=0.7, random_state=7565)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3 = knn3.fit(X=train_inputs, y=train_classes)
knn3_predictions = knn3.predict(test_inputs)
knn3_score = knn3.score(X=test_inputs, y=test_classes)
print(f"The accuracy of the KNN3 classifier: {round(knn3_score * 100, 2)}%")

knn3_matrix = confusion_matrix(y_true=test_classes, y_pred=knn3_predictions)
print(knn3_matrix)

knn5 = KNeighborsClassifier(n_neighbors=5)
knn5 = knn5.fit(X=train_inputs, y=train_classes)
knn5_predictions = knn5.predict(test_inputs)
knn5_score = knn5.score(X=test_inputs, y=test_classes)
print(f"The accuracy of the KNN5 classifier: {round(knn5_score * 100, 2)}%")

knn5_matrix = confusion_matrix(y_true=test_classes, y_pred=knn5_predictions)
print(knn5_matrix)

knn11 = KNeighborsClassifier(n_neighbors=11)
knn11 = knn11.fit(X=train_inputs, y=train_classes)
knn11_predictions = knn11.predict(test_inputs)
knn11_score = knn11.score(X=test_inputs, y=test_classes)
print(f"The accuracy of the KNN11 classifier: {round(knn11_score * 100, 2)}%")

knn11_matrix = confusion_matrix(y_true=test_classes, y_pred=knn11_predictions)
print(knn11_matrix)

naive_bayes = GaussianNB()
naive_bayes.fit(X=train_inputs, y=train_classes)
naive_bayes_predictions = naive_bayes.predict(test_inputs)
naive_bayes_score = naive_bayes.score(X=test_inputs, y=test_classes)
print(f"The accuracy of the naive-bayes classifier: {round(naive_bayes_score * 100, 2)}%")

bayes_matrix = confusion_matrix(y_true=test_classes, y_pred=naive_bayes_predictions)
print(bayes_matrix)
