import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# lay du lieu tu thu vien
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# gan nhan cho label
X0 = iris_X[iris_y == 0, :]
X1 = iris_X[iris_y == 1,:]
X2 = iris_X[iris_y == 2,:]

# tach training va test
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights='distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# print ("Print results for 20 test data points:")
# print ("Predicted labels: ", y_pred[20:40])
# print ("Ground truth    : ", y_test[20:40])

print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))