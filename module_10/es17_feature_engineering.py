# FEATURE ENGINEERING
# 17) Prendi un dataset e applica tecniche di feature engineering, come la creazione di nuove feature o la trasformazione delle feature esistenti. Valuta come queste modifiche influenzano le prestazioni del tuo modello di machine learning.

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    data = fetch_california_housing()       # dati delle case del california
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def baseline_model(df):
    X = df.drop(columns='target')       # tolgo le label da x
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)    # divido il dataset
    
    scaler = StandardScaler()       # applico lo scaler per rimuovere la media e avere varianza unitaria
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()      # regressione lineare
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def feature_engineered_model(df):
    df = df.copy()                                              # copio il df

    df["Inc_Occ_Interaction"] = df["MedInc"] * df["AveOccup"]   # creo una nuova feature

    df["LogPopulation"] = np.log1p(df["Population"])            # trasformazione logaritmica della popolazione

    df["OccupancyLevel"] = pd.cut(df["AveOccup"], bins=[0, 2, 5, 15, np.inf], labels=[0, 1, 2, 3])
    # raggruppa diversi intervalli numerici in bin o bucket

    df = pd.get_dummies(df, columns=["OccupancyLevel"], drop_first=True) # faccio encode delle feature (vengono trasformate in numero)

    X = df.drop(columns='target')
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    return mse

if __name__ == "__main__":
    df = load_data()
    mse_base = baseline_model(df)
    mse_feat = feature_engineered_model(df)

    print(f"MSE base model: {mse_base:.3f}")
    print(f"MSE con feature engineering: {mse_feat:.3f}")


'''
FEATURE ENGINEERING
Il processo di trasformazione dei dati grezzi in caratteristiche (features) utili per i modelli di machine learning.
L'obiettivo è migliorare le performance predittive del modello creando, selezionando, o trasformando variabili in modo da
rappresentare meglio il fenomeno che si vuole modellare.
- creazione di nuove feature
- trasformazione di feature esistenti
* Log transformation
* Interazione tra feature: MedINc x AceOccup (reddito * densità)
Binning: trasformazione continua -> categoria
Polynomial features

Tecniche Standard
Pulizia dei dati
    - Rimozione dei valori nulli o imputazione (media, mediana, moda, KNN)
    - correzione di outlier o codifica coerente. Sono data point molto diversi dal resto
Codifica delle variabili categoriche
    - Label Encoding (per variabili ordinali): hanno un ordinamento, come i voti A+ A B+ B
    - One-Hot Encoding (per variabili nominali): dati categorici senza alcun ordine coem stati o rami. Possono essere convertiti in dati numerici
    - Target Encoding 
Normalizzazione e Scaling
    - MinMax Scaling (Normalization): processo di scalare le feature tra 0 e 1. Per dati che non seguono una distribuzione gaussiana come SVM e decision tree
                attenzione che SVM con kernel rbf ha bisogna dello z-score
    - Standard Scaling (Z-score): processo di sottrarre la media da ogni feature e dividere tutto per la deviazione standard
                permette di avere ogni feature con media 0 e deviazione standard 1
    - Robust Scaling (meno sensibile agli outlier)
Estrazione di feature da date
    - estrazione di anno, mese, giorno, giorno della settimana, ora
    - calcolo di intervalli di tempo tra eventi
Interazioni e combinazioni
    - creazione di feature combinatorie, ad esempio feature_a * feature_b ... feature_a / feature_b
Aggreazioni
    - per dati temporali basati su gruppi: media, max, min, count

TECNICHE AVANZATE
Tecniche statistiche
    - PCA (Principal Component Analysis): per ridurre la dimensionalità mantenendo la varianza
    - TSNE / UMAP: per embedding non lineari (usate anche per la visualizzazione cluster)
    - Feature polinomiali: costruzione di feature polinomiali di ordine superiore.
Embedding e rappresentazioni dense
    - Word Embedding (testo): Word2Vec, Glove, fasttext
    - Entity Embedding: encoding denso per variabili categoriche (spesso con reti neurali)
    - Autoencoder: per compressione e riconstruzione, utile per estrarre rappresentazioni latenti
Feature selection automatica
    - Lasso Regression (L1)
    - Random Forest Feature Importance
    - Recursive Feature Elimination (RFE)
    - SHAP
Tecniche di feature engineering specifiche per dominio
    - NLP: n-gram, tf-idf, sentimnet score, pos tagging
    - immagini: estrazine di edge, texture, istogrammi, CNN-based features
    - serie temporali: rolling mean, differenziali, Fourier/wavelet transform
Feature extraction automatica

'''