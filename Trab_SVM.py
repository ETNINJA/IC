from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets		# To Get iris dataset
from sklearn import svm    			# To fit the svm classifier
import numpy as np                  # manipular dados
import pandas as pd
import matplotlib.pyplot as plt     # To plotar

iris = datasets.load_iris()

# Create feature and target arrays
X = iris.data[:,:3]
y = iris.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

print(X_train)
print()
print(X_test)
print()
print(y_test)


classifier = svm.SVC(kernel='linear', C=1.0,gamma= 'auto').fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Print the accuracy
print("Accuracy",classifier.score(X_test, y_test))

# Making the Confusion Matrix QUANTIDADE DE ERROS
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print()
print("Confusion Matrix\n", cm)