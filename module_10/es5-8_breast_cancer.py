# BREAST CANCER
# 5) Carica il dataset Breast Cancer incluso in scikit-learn e visualizza le prime 5 righe dei dati.
# 6) Dividi il dataset Breast Cancer in un set di addestramento (70%) e un set di test (30%).
# 7) Crea un modello di regressione utilizzando scikit-learn, ad esempio un regressore lineare, e addestralo con il set di addestramento.
# 8) Valuta il modello di regressione creato nel punto 7 utilizzando il set di test e calcola l'errore quadratico medio (Mean Squared Error).

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

def load_data():
    print("Carico il Breast Cancer dataset e visualizzo le prime 5 righe")
    bc = load_breast_cancer()
    x = pd.DataFrame(bc.data, columns=bc.feature_names)
    y = pd.Series(bc.target, name='target')
    print(x.head(5))
    return x, y

def train_test(x, y):
    print("Divisione dei dati di addestramento (70%) e test (30%)")
    return train_test_split(x, y, test_size=0.3)

def regression_model(x_train, y_train):
    print("Addestramento del modello di regressione lineare")
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def evaluation_mse(model, x_test, y_test):
    print("Valutazione del modello - Calcolo del Mean Squared Error")
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse:.4f}")
    return mse

if __name__ == '__main__':
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test(x, y)
    model = regression_model(x_train, y_train)
    evaluation_mse(model, x_test, y_test)