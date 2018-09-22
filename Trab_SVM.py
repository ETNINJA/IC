# **************************************************
# Practice 01  for Computational Intelligence
# Teacher: Luciana Balieiro
# Team: Anne Almeida, Geovane Richards e Robert 
# Algorithm: SVM (Support Vector Machine) 
# **************************************************

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets		# To Get iris dataset
from sklearn import svm    			# To fit the svm classifier
import numpy as np                  # manipular dados
import matplotlib.pyplot as plt     # To plot

iris = datasets.load_iris()

# Create feature and target arrays (Petal lenght and width)
X = iris.data[:,2:4]
y = iris.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

#Using the classifier and fitting the data into the model
classifier = svm.SVC(kernel='linear', C=1.0,gamma= 'auto').fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Printing the accuracy
print("\n*Accuracy\n",classifier.score(X_test, y_test))

# Printing the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("\n*Confusion Matrix\n", cm)

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],  c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('SVM - linear (Petal)')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend()
plt.show()
