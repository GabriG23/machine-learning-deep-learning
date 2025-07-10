# SUPPORTO VECTOR MACHINE
# 19 Crea un modello di classificazione utilizzando Support Vector Machines (SVM) da scikit-learn e addestralo.
# 20 Valuta le prestazioni del modello di classificazione SVM creato nell’esercizio 19 utilizzando la
# cross-validation e calcolando l'accuratezza media.

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def load_classification_data():
    X, y = make_classification(n_samples=500, n_features=10, n_informative=6, 
                               n_redundant=2, n_classes=2, random_state=42)
    return X, y

def svm_model(X, kernel='rbf'):
    scaler = StandardScaler()   # rimuove la media e
            # e scala tutto a varianza unitaria
            # z = (x - u) / s
            # questo perché il svm con kernel richiede che la i valori siano tutti centrati e la varianza scalata
            # perché se c'è una feature con varianza maggiore delle altre tenderà ad influenzare di più il modello
    X_scaled = scaler.fit_transform(X) # mean = 0, std = 1
    svm = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
    # C è il parametro di regolarizzazione (maggiore C = meno tolleranza agli errori)
    # gamma = scale, definisce l'influenza di un singolo punto, scale si basa sulla varianza
    # svm con kernel rbf, quindi per dati non linearmente separabili
    return svm, X_scaled

def evaluation(svm, X_scaled, y):
    scores = cross_val_score(svm, X_scaled, y, cv=5, scoring='accuracy')
    # ritorna 5 valori di accuratezza per ciascun fold, ed infine calcola la media
    print("Accuratezza per ciascun fold:")
    print(scores)
    print(f"\nAccuratezza media: {scores.mean():.2f}")

if __name__ == '__main__':
    X, y = load_classification_data()
    model, X_scaled = svm_model(X, kernel='rbf')
    evaluation(model, X_scaled, y)

'''
SVM (Support Vector Machine)
Le support vector machines sono modelli di apprendimento supervisionato che cercano di trovare
l'iperpiano ottimale che separa due classi con il margine più ampio possibile.
Gli SVM lineari trovano separazioni lineari; le SVM con kernal possono separare
- support vector: punti dati più vicini al margine
- margine: distanza tra l'iperpiano e i support vector
- kernel: funzione che trasforma i dati per renderli separabili
- cross validation: è una tecnica per valutare il modello suddividendo il dataset in più partizioni (folds).
Ad ogni iterazione si usa una porzione per test e il resto per training.
Vantaggio: stima più robusta delle prestazioni generali del modello
'''