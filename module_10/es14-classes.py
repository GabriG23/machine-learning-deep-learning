# BINARY VS MULTICLASS CLASSIFICATION
# 14) Sperimenta la differenza tra problemi di classificazione binaria e multiclasse utilizzando un dataset appropriato.
# Addestra un classificatore binario e un classificatore multiclasse sullo stesso dataset e confronta i risultati.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

def load_data():
    iris = load_iris()
    x = iris.data
    y = iris.target
    target_names = iris.target_names
    return x, y, target_names

def binary_classification(x, y):
    print("\nClassificazione binaria: setosa vs non-setosa")
    y_binary = np.where(y == 0, 1, 0)  # Etichetta binaria: 1 se è 'setosa', 0 altrimenti

    x_train, x_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2)
    clf = SVC() # kernel = rbf se non specifico niente
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print("Accuracy (binary):", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["non-setosa", "setosa"]))

def multiclass_classification(x, y, target_names):
    print("\n== Classificazione multiclasse: setosa vs versicolor vs virginica ==") # abbiamo 3 classi
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print("Accuracy (multiclass):", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == '__main__':
    x, y, target_names = load_data()
    binary_classification(x, y)
    multiclass_classification(x, y, target_names)
