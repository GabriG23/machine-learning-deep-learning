import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_df(path):
    # 1 Scrivi un programma che acquisisca dati da un file CSV chiamato "dati.csv" e li carichi in un DataFrame Pandas.
    print("1. Lettura dati")
    return pd.read_csv(path)

def missing_values(df):
    # 2 Supponiamo che nel DataFrame "data" ci siano valori mancanti. Scrivi un programma che identifichi
    # e gestisca i valori mancanti sostituendoli con la media dei valori presenti nella stessa colonna.
    print("2. Sostituzione dei valori mancanti\n")
    new_df = df.fillna(df.mean(numeric_only=True))
    return new_df

def plot_data(df):
    # 3 Utilizzando Matplotlib, crea un grafico a dispersione (scatter plot) dei dati presenti in due colonne del DataFrame "data".
    print("3. Stampa delle colonne Length e Width.")
    x = df['SepalLengthCm']
    y = df['SepalWidthCm']

    plt.scatter(x, y)
    plt.show()

def array_bi():
    # 4 Crea un array Numpy bidimensionale con valori casuali di dimensione 3x3.
    print("4. Creazione array casuale")
    rand_arr = np.random.rand(3,3)
    print(rand_arr)
    return

def array_mul():
    # Dato un array Numpy A (3x3) e un array B (3x2), scrivi un programma che
    # esegua la moltiplicazione tra di essi e stampi il risultato.
    print("5. Prodotto tra due array")
    arr1 = np.random.rand(3, 3)
    arr2 = np.random.rand(3, 2)
    prod = np.dot(arr1, arr2)
    print(prod)
    return

def avg_std(df):
    # 6 Calcola la media e la deviazione standard delle colonne del DataFrame "data" utilizzando Pandas.
    print("6. Calcolo media e deviazione standard")
    for col in df.columns:  # scorro colonne
        if df[col].apply(lambda x: isinstance(x, float)).all():
            print(f"Avg per {col}: {df[col].mean()}")
            print(f"Std per {col}: {df[col].std()}")
        else:
            print(f"La colonna {col} non ha numeri float")

def train_val(df):
    # 7 Dividi il DataFrame "data" in due sottoinsiemi: uno per l'addestramento del modello (80% dei dati)
    # e uno per la validazione (20% dei dati).
    print("Divisione del df in dati di train e validation")
    # il mio target sono le species (classi) del dataset
    X_train, X_val, y_train, y_val = train_test_split(df.drop(columns='Species'), df['Species'], train_size=0.80, random_state=42)
    return X_train, X_val, y_train, y_val

def linear_regression():
    # 8 Supponiamo di avere due array Numpy "X" e "y" che rappresentano le feature e i target
    # di un modello di regressione. Crea un modello di regressione lineare utilizzando scikit-learn e addestralo.
    print("Addestramento Linear Regression con due Numpy array")
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression().fit(X, y)
    return reg

def coeff_r():
    # 9 Valuta le prestazioni del modello di regressione creato nel punto 8 utilizzando il coefficiente di determinazione R².
    print("Valutazione del modello di regressione - R^2")
    # stessi dati dell'esercizio 8
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3  # y = 1*x1 + 2*x2 + 3

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    print(f"R2 Score: {r2:.2f}")

def normalization(df):
    # 10 Supponiamo che tu abbia un array Numpy "data" con dati numerici.
    # Scrivi un programma che utilizzi scikit-learn per normalizzare i dati in modo che abbiano una media zero e una deviazione standard uno.
    print("Normalizzazione dei dati")
    numeric_columns = df.select_dtypes(include=['float']).columns    
    scaler = StandardScaler() # questo per avere deviazione standard di 1
    scaler_min_max = MinMaxScaler() # questo per mettere i dati tra 0 e 1

    normalized = scaler.fit_transform(df[numeric_columns])
    normalized_df = pd.DataFrame(normalized, columns=numeric_columns)

    normalized_min_max = scaler_min_max.fit_transform(df[numeric_columns])
    normalized_df_min_max = pd.DataFrame(normalized_min_max, columns=numeric_columns)
    
    print("Dati normalizzati:")
    print(normalized_df.head())

    print("Dati normalizzati tra 0 e 1:")
    print(normalized_df_min_max.head())


if __name__ == '__main__':
    df = load_df("../dataset/Iris.csv") # 1
    df = missing_values(df) # 2
    print(df.columns)
    # plot_data(df) # 3
    # array_bi() # 4
    # array_mul() # 5
    # avg_std(df) # 6
    # _, _, _, _ = train_val(df) # 7
    # _ = linear_regression(df) # 8
    coeff_r()
    normalization(df)

# StandardScaler è una tecnica di scaling che standardizza le feature
# rimuovendo la media e scalandole a varianza unitaria.
# questo è utile negli algoritmi di ML per algoritmi sensibili alla scala dei dati di input
# come SVM o PCA
# Mean Removal: centra i dati sottraendo la media di ogni feature
# Unit Variance Scaling: Divide ogni feature dalla sua deviazione standard,
#   permettendo una deviazione standard unitaria di 1

# MinMaxScaler è una tecnica di scaling che trasforma le feature in un range specifico. Questo è
# utile quando le feature hanno range diversi, e permette a tutte le feature di contribuire al modello


'''
StandardScaler
Standardize features by removing the mean and scaling to unit variance.
The standard score of a sample x is calculated as:

    z = (x - u) / s

where u is the mean of the training samples or zero if with_mean=False, and s is the standard deviation of the training samples or one if with_std=False.

Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set.
Mean and standard deviation are then stored to be used on later data using transform.
Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features
do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).
For instance many elements used in the objective function of a learning algorithm (such as the RBF kernel of Support Vector Machines or
the L1 and L2 regularizers of linear models) assume that all features are centered around 0 and have variance in the same order.
If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective
function and make the estimator unable to learn from other features correctly as expected.

StandardScaler is sensitive to outliers, and the features may scale differently from each other in the presence of outliers. For an example visualization, refer to Compare StandardScaler with other scalers <plot_all_scaling_standard_scaler_section>.

This scaler can also be applied to sparse CSR or CSC matrices by passing with_mean=False to avoid breaking the sparsity structure of the data.



MinMaxScaler
Transform features by scaling each feature to a given range.
This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.

The transformation is given by:

    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min

where min, max = feature_range.

This transformation is often used as an alternative to zero mean, unit variance scaling.
MinMaxScaler doesn't reduce the effect of outliers, but it linearly scales them down into a fixed range,
where the largest occurring data point corresponds to the maximum value and the smallest one corresponds
to the minimum value.
For an example visualization, refer to Compare MinMaxScaler with other scalers <plot_all_scaling_minmax_scaler_section>.


'''