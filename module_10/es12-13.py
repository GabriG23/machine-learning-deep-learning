# REGRESSIONE + GRID_SEARCH

#12) Carica un dataset di regressione (ad esempio, il dataset delle case di Boston) e sperimenta diversi
# algoritmi di regressione come regressione lineare, regressione polinomiale e regressione Ridge.
# Valuta e confronta le prestazioni di ciascun algoritmo.

#13) Prendi uno dei dataset utilizzati negli esercizi 11 o 12 e utilizza la ricerca degli iperparametri
# per trovare i migliori parametri per uno degli algoritmi di machine learning.
# Ad esempio, puoi utilizzare GridSearchCV per ottimizzare gli iperparametri di una regressione Ridge.

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

def load_data():
    print("Caricamento dataset California Housing")
    data = fetch_california_housing()
    x = data.data
    y = data.target
    return train_test_split(x, y, test_size=0.2)

def train_regression_models(x_train, x_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Polynomial Regression (deg=2)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        "Ridge Regression": Ridge(alpha=1.0)
    }

    results = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = mse
        print(f"{name}: MSE = {mse:.4f}")

    return results

def grid_search_ridge(x_train, y_train):
    print("\nRicerca dei migliori iperparametri per Ridge Regression con GridSearchCV...")
    param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100]
    }
    grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid.fit(x_train, y_train)
    print("Miglior alpha trovato:", grid.best_params_['alpha'])
    print("Miglior MSE (negativo):", grid.best_score_)
    return grid.best_estimator_

if __name__ == '__main__':

    x_train, x_test, y_train, y_test = load_data()      # Caricamento e divisione dati

    scaler = StandardScaler()                           # Standardizzazione
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    results = train_regression_models(x_train, x_test, y_train, y_test) # confronto modelli di regressione

    best_ridge = grid_search_ridge(x_train, y_train)    # GridSearchCV su Ridge

    y_pred = best_ridge.predict(x_test)                 # Valutazione finale su test set
    final_mse = mean_squared_error(y_test, y_pred)
    print(f"\nMSE finale del miglior Ridge sul test set: {final_mse:.4f}")
