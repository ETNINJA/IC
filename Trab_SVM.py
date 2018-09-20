#Anne Almeida, Geovane Richards e Robert
# SVM - acurácia e matrix de confusão

# bilbis
from sklearn import datasets		# To Get iris dataset
from sklearn import svm    			# To fit the svm classifier
import numpy as np                  # manipular dados
import pandas as pd
import matplotlib.pyplot as plt     # To plotar


# import iris dataset
iris = datasets.load_iris()

# *****************PARA ENTENDER O DATABASE

# print na description
print ("Iris data set Description :: ", iris['DESCR'])

#print os dados
print ("Iris feature data :: ", iris['data'])

# Classifica 0 1 2 (setosa , versicolour e virginica)
print ("Iris target TODOS :: ", iris['target'])
print()
print ("setosa", iris.target[:50])
print()
print ("versicolour", iris.target[50:100])
print()
print ("virginica", iris.target[100:150])

#***************** FAZENDO O SPLIT 40 para o training and 10 for the test

X = iris.data[:, :4]  # todas as features.
y = iris.target
print ("teste", y)

X_trainS = iris.data[:40, 0:3]
X_testS = iris.target[40:50]
y_trainS = iris.data[:40, 0:3]
y_testS = iris.target[40:50]
print("y_test Setosa", y_testS)

X_trainV = iris.data[50:90, 0:3]
X_testV = iris.target[90:100]
y_trainV = iris.data[50:90, 0:3]
y_testV = iris.target[90:100]
print("y_test Versicolour", y_testV)

X_trainVi = iris.data[100:140, 0:3]
X_testVi = iris.target[140:150]
y_trainVi = iris.data[100:140, 0:3]
y_testVi = iris.target[140:150]
print("y_test Virginica", y_testVi)

X_train = iris.data[100:140, 0:3]
X_test = iris.target[140:150]
y_train = iris.data[100:140, 0:3]
y_test = np.array([y_testS, y_testV, y_testVi])
print("y_test CONCATENADOS", y_test)





#***************************SVM daqui p baixo **************************

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
