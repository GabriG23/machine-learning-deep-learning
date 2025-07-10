# KMEANS
# 18) Carica un dataset adatto al clustering (come Iris) e sperimenta diversi algoritmi di clustering, come il clustering K-Means, il clustering gerarchico e il DBSCAN. Valuta e confronta i risultati.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

def load_data():        # lettura dati
    data = load_iris()
    x = data.data
    y = data.target     # non dovrebbero servirci, il clustering è unsupervised
    return x, y

def run_clustering(x, y):
    models = {  # abbiamo 3 tipi di clustering
        # sappiamo che abbiamo 3 classi, quindi 3 cluster
        "KMeans": KMeans(n_clusters=3), # attenti che cade in un minimo locale
        "Agglomerative": AgglomerativeClustering(n_clusters=3),     # assegna un cluster ad ogni punto
        "DBSCAN": DBSCAN(eps=0.6, min_samples=4)        # con densità (usa il raggio)
    }

    results = {}

    for name, model in models.items():
        preds = model.fit_predict(x)
        sil = silhouette_score(x, preds)
        ari = adjusted_rand_score(y, preds)
        results[name] = {"labels": preds, "silhouette": sil, "ARI": ari}
        print(f"{name} → Silhouette: {sil:.3f} | ARI: {ari:.3f}")

    return results

def plot_clusters(x, results):
    plt.figure(figsize=(15, 4))
    for i, (name, res) in enumerate(results.items(), 1):
        plt.subplot(1, 3, i)
        sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=res["labels"], palette="Set2")
        plt.title(f"{name} clustering")
        plt.xlabel("Sepal length")
        plt.ylabel("Sepal width")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    x, y = load_data()
    x = StandardScaler().fit_transform(x)  # Scaling consigliato per DBSCAN
    results = run_clustering(x, y)
    plot_clusters(x, results)



'''
Kmeans
- In practice, the k-means algorithm is very fast (one of the fastest clustering algorithms available), but it falls in local minima. That's why it can be useful to restart it several times.
- divide i dati in k cluster cercando di minimizzare la varianza intra-cluster
    - sceglie k
    - inizializza casualmente k centroidi
    - assegna ogni punto al centroide più vicino
    - ricalcola i centroidi
    - ripete finché i centroidi non cambiano significativamente
    pro: veloce su grandi dataset, facile
    contro: scelta di k, sensibile agli outlier, raggiunge spesso un minimo locale

DBSCAN (Density-Based Spatial Clustering of Applications with Nois)
- trova cluster in base alla densità di punti. I punti in aree dense sono clusterizzati, quelli isolati sono considerati rumore.
- definisce due parametri:
    * eps (raggio) e min_samples
    * un punto è un "core point" se ha almeno min_samples vicini entro eps
    * costruire un cluster attorno ai core points
    * i punti che non appartengono a nessun cluster sono etichettati come rumore
    pro: non serve k, trova cluster, buono con outlier
    contro: sensibile a eps e min_samples, poco efficace con densità molto variabili, poco scalabile

Agglomerative Clustering
Costruisce una gerarchia di cluster unendo ricorsivamente i due più vicini
    - ogni punto è un cluster
    - unisce i due cluster più vicini (secontro un criterio di linkage)
    - ripete finché non sia arriva a k cluster
    pro: non assume forma dei cluster, crea un dendrogramma utile per analisi esplorativa
    contro: complessità computazionale, non gestisce bene dati rumorosi, devi decidere dove tagliare il dendrogramma


Metriche:
Silhoutte_score (valido senza label, quindi unsupervised)
- misura quanto ogni punto è vicino al suo cluster rispetto agli altri cluster
- range da -1 a 1
    1 punti perfettamente nel proprio cluster
    0 punti sui confini
    -1 punti assegnati al cluster sbagliato
- serve a valutare la coesione e separazione dei cluster

ARI (Adjusted Rand Index) valido per supervised
- confronta i cluster predetti con i gruppi veri (quindi il target)
- range da -1 a 1
    1 clustering perfetto
    0 clustering casuale
    <0 peggio del caso random 
- è una metrica supervisionata (usata qui solo perché iris ha le etichette, anche se il clustering è unsupervised)
'''