# K-FOLD CROSS VALIDATION

# 15) Utilizza la K-Fold Cross-Validation per valutare le prestazioni di un algoritmo di classificazione o regressione su un dataset.
# Calcola la media delle misure di prestazione (ad esempio, accuratezza per la classificazione o errore quadratico medio
# per la regressione) su diverse ripartizioni del dataset.

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.datasets import fetch_california_housing

def kfold_classification():
    print("K-Fold Cross Validation: Classificazione con SVM")

    iris = load_iris()
    x = iris.data
    y = iris.target

    model = SVC()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # K-Fold cross-validator.
    # Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
    # Each fold is then used once as a validation while the k - 1 remaining folds form the training set.

    scores = cross_val_score(model, x, y, cv=kf, scoring='accuracy')

    print("Accuratezze per ogni fold:", scores)
    print(f"Media accuratezza: {np.mean(scores):.2f}")
    print(f"Deviazione standard: {np.std(scores):.2f}")

def kfold_regression():
    print("K-Fold Cross Validation: Regression con Ridge")
    x, y = fetch_california_housing(return_X_y=True)
    model = Ridge()

    scores = cross_val_score(model, x, y, cv=5, scoring='neg_mean_squared_error')
    mse_scores = -scores
    print("MSE per ogni fold:", mse_scores)
    print(f"MSE medio: {np.mean(mse_scores)}")


if __name__ == '__main__':
    kfold_classification()
    kfold_regression()
