# PCA
# 15 Applica la riduzione della dimensionalità su un set di dati "X" utilizzando la PCA (Principal Component Analysis) da scikit-learn.
# 16 Crea un grafico dei dati ridotti dopo aver applicato la PCA. Ogni punto nel grafico rappresenterà una riga in "X_reduced".
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data():
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, n_classes=2, random_state=42)
    # genera dati sintetici per un problema di classificazione binaria
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), y


def pca_model(X):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    print(f"Forma originale di X: {X.shape}")
    print(f"Forma dopo PCA: {X_reduced.shape}")
    return X_reduced

def pca_graph(X_reduced):
    # Esercizio 16: Grafico dei dati ridotti
    plt.figure(figsize=(8, 6))
    plt.grid()
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.7)
    plt.xlabel('Prima componente principale')
    plt.ylabel('Seconda componente principale')
    plt.title('Dati proiettati sulle prime 2 componenti principali (PCA)')
    plt.colorbar(label='Classe')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    X, y = load_data()
    X_reduced = pca_model(X)
    pca_graph(X_reduced)


'''
PCA (PRINCIPAL COMPONENT ANALYSIS)
La PCA è una tecnica di riduzione della dimensionalità. Serve per proiettare dati ad alta dimensione in uno 
spazio di dimensioni inferiori, conservando il massimo della varianza
- si calcolano prima le componenti principali, cioè combinazioni lineari delle feature originali che spiegano la maggior parte della varianza nei dati.
- 
'''