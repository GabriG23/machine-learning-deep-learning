# CLASSIFICATION + GRIDSEARCH
# 11) Carica un dataset di classificazione (come Iris o Breast Cancer) e sperimenta diversi algoritmi di
# classificazione disponibili in scikit-learn, come Support Vector Machine (SVM), Random Forest, e k-Nearest Neighbors (k-NN).
# Valuta e confronta le prestazioni di ciascun algoritmo.

#13) Prendi uno dei dataset utilizzati negli esercizi 11 e utilizza la ricerca degli iperparametri per trovare
# i migliori parametri per uno degli algoritmi di machine learning. Ad esempio, puoi utilizzare GridSearchCV per ottimizzare
# gli iperparametri di un modello SVM.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd

def load_iris_data():
    print("Carico il dataset Iris")
    iris = load_iris()
    x = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="Species")
    print(x.head())
    return x, y

def train_models(x_train, x_test, y_train, y_test):

    print("\nConfronto Modelli")

    models = {
        "SVM (linear)": SVC(kernel='linear'),
        "SVM (rbf)": SVC(kernel='rbf'),
        "SVM (poly)": SVC(kernel='poly', degree=3),
        "SVM (sigmoid)": SVC(kernel='sigmoid'),
        "Random Forest": RandomForestClassifier(),
        "k-NN": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = {}

    print("== Valutazione modelli ==")
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name}: Accuracy = {acc:.4f}")

    return results

def svm_grid_search(x_train, y_train):
    print("\nEseguo GridSearchCV su SVM...")
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    grid = GridSearchCV(SVC(), param_grid, cv=5)
    grid.fit(x_train, y_train)
    print("Migliori parametri trovati:")
    print(grid.best_params_)
    print(f"Accuratezza migliore nel cross-validation: {grid.best_score_:.4f}")
    return grid.best_estimator_


if __name__ == '__main__':

    x, y = load_iris_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    scaler = StandardScaler()  # Standardizzazione (facoltativa ma utile per SVM e k-NN)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    results = train_models(x_train, x_test, y_train, y_test)    # Addestramento dei modelli
    best_svm = svm_grid_search(x_train, y_train)                # Ottimizzazione iperparametri con GridSearchCV su SVM

    y_pred = best_svm.predict(x_test)                   # 5. Valutazione finale sul test set
    final_acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuratezza finale del miglior SVM sul test set: {final_acc:.4f}")

    print("\n== Riepilogo modelli ordinati per accuratezza ==")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for name, acc in sorted_results:
        print(f"{name}: {acc:.4f}")