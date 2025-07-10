# Decision Tree
# 17 Crea un modello di regressione utilizzando alberi decisionali da scikit-learn e addestralo.
# 18 Visualizza l'albero decisionale creato nell’esercizio 17 utilizzando la libreria "graphviz".

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import graphviz

def load_regression_data():
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    # genera dati sintetici per un problema di regressione (valori continui)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), y

def decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg_tree = DecisionTreeRegressor(max_depth=4, random_state=42)
    reg_tree.fit(X_train, y_train)

    y_pred = reg_tree.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    return reg_tree

def plot_graph(reg_tree):
    dot_data = export_graphviz(
        reg_tree,
        out_file=None,
        feature_names=X.columns,
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree_regressor", format="png", cleanup=False)
    graph.view()

if __name__ == '__main__':
    X, y = load_regression_data()
    tree = decision_tree(X, y)
    plot_graph(tree)

'''
Decision Tree:
albero decisionale per regressione che suddivide iterativamente il dataset in sottoinsiemi
omogenei rispetto al target numerico.
A ogni nodo, l'albero sceglie la feature e il valore di split che minimizza l'errore quadratico medio (MSE)
- il parametro max_depth limita la profondità dell'albero per evitare overfitting

GRAPHVIZ
- export_graphviz esporta l'albero in formato .dot
ogni noda mostra
- il criterio di split, il valor medio del nodo target, il numero di campioni, la varianza residua
'''