# Logistic Regression
# 11 Supponiamo di avere un set di dati con feature "X" e target "y" per un problema di classificazione binaria.
# Crea un modello di classificazione logistica utilizzando scikit-learn e addestralo.
# 12 Valuta le prestazioni del modello di classificazione creato nell’esercizio 
# utilizzando la matrice di confusione e il punteggio di precisione.
# 13 Utilizza il modello creato nell’esercizio 11 per estrarre le feature più importanti nel problema di classificazione.
# 14 Crea un grafico a barre che mostri le feature importanti estratte nel’esercizio 13.

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, n_classes=2, random_state=42)
    # genera dati sintetici per un problema di classificazione binaria
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), y

def model_evaluation(X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred) # mostra il numero di veri positivi, veri negativi, falsi positivi e falsi negativi
    precision = precision_score(y_test, y_pred) # predizioni positive corrette

    print("Matrice di confusione:")
    print(cm)
    print(f"Precisione: {precision:.2f}")

def important_feature(model, X):
    feature_importance = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
    # la logistic regression assegna un coefficiente ad ogni feature
    # più è alto il valore più importante è la feature nel determinare la classe
    print("\nFeature più importanti:")
    print(feature_importance)
    return feature_importance

def plot_data_bar(feature_importance):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index)
    plt.title("Importanza delle feature (modello di regressione logistica)")
    plt.xlabel("Coefficiente (importanza)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    model_evaluation(X_test, y_test)

    feature_importance = important_feature(model, X)

    plot_data_bar(feature_importance)



'''
Cosa fa make_classification?
Generate a random n-class classification problem.

This initially creates clusters of points normally distributed (std=1) about vertices of an n_informative-dimensional hypercube with sides of length 2*class_sep and assigns an equal number of clusters to each class. It introduces interdependence between these features and adds various types of further noise to the data.

Without shuffling, X horizontally stacks features in the following order: the primary n_informative features, followed by n_redundant linear combinations of the informative features, followed by n_repeated duplicates, drawn randomly with replacement from the informative and redundant features. The remaining features are filled with random noise. Thus, without shuffling, all useful features are contained in the columns X[:, :n_informative + n_redundant + n_repeated].


'''