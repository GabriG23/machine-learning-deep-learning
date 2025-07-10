# IRIS-DATASET
# 1) Carica il dataset Iris incluso in scikit-learn e visualizza le prime 5 righe dei dati.
# 2) Dividi il dataset Iris in un set di addestramento (80%) e un set di test (20%).
# 3) Crea un modello di classificazione utilizzando scikit-learn, ad esempio un classificatore SVM,
#   e addestralo con il set di addestramento.
# 4) Valuta il modello di classificazione creato nell’esercizio 3 utilizzando il set di test e calcola l'accuratezza.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

def load_iris_data():
    print("Carico l'iris dataset e visualizzo le prime 5 righe")
    iris = load_iris()
    x = iris.data
    y = iris.target
    df_train = pd.DataFrame(x, columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])
    df_test = pd.DataFrame(y, columns=["Species"])
    print(df_train.head(5))
    return df_train, df_test

def train_test(df_train, df_test):
    print("Divisione dei dati in addestramento e testing")
    return train_test_split(df_train, df_test, test_size=0.2)

def model_svm(x_train, y_train):
    print("Addestramento del modello SVM linear")
    mod = SVC(kernel='linear')
    mod.fit(x_train, y_train)
    return mod

def evaluation(model, x_test, y_test):
    print("Valuazione del modello")
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuratezza del modello: {acc:.2f}")

if __name__ == '__main__':
    train_data, test_data = load_iris_data()
    xtr, xte, ytr, yte = train_test(train_data, test_data)
    model = model_svm(xtr, ytr)
    evaluation(model, xte, yte)