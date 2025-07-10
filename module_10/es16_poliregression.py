# 16) Crea un modello di regressione polinomiale su un dataset di regressione
# e sperimenta l'effetto di vari gradi di polinomio sul fenomeno dell'overfitting e dell'underfitting.
# Visualizza le curve di apprendimento e confronta i risultati.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

def load_data():    # prendo i dati del california housing
    data = fetch_california_housing()
    x = data.data[:, [0]]  # MedInc (reddito mediano)
    y = data.target        # Prezzo medio casa
    return x, y

def polynomial_regression_demo(degrees=[1, 3, 10]):
    x, y = load_data()
    x = x[:5000]
    y = y[:5000]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    x_range = np.linspace(x.min(), x.max(), 500).reshape(-1, 1) # Return evenly spaced numbers over a specified interval.

    plt.figure(figsize=(18, 4))

    for i, deg in enumerate(degrees, 1):
        model = make_pipeline(
            StandardScaler(),                   # <--- scalatura importante!
            PolynomialFeatures(degree=deg),
            LinearRegression()
        )
        model.fit(x_train, y_train)

        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        y_plot = model.predict(x_range)

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        plt.subplot(1, len(degrees), i)
        plt.scatter(x_train, y_train, color='lightgray', alpha=0.5, label="Train data")
        plt.plot(x_range, y_plot, color='red', label=f"Model (deg={deg})")
        plt.title(f"Degree {deg}\nTrain MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")
        plt.xlabel("MedInc (scaled)")
        plt.ylabel("MedHouseVal")
        plt.legend()

    plt.tight_layout()
    plt.suptitle("Regressione polinomiale (con preprocessing)", fontsize=16, y=1.05)
    plt.show()


if __name__ == "__main__":
    polynomial_regression_demo([1, 3, 10])



