# Dataset
# 9) Carica un dataset personalizzato in formato CSV utilizzando Pandas e crea un modello di classificazione utilizzando scikit-learn.
# 10) Valuta il modello di classificazione creato nell’esercizio 9 utilizzando il dataset personalizzato e calcola l'accuratezza.
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_healthcare(filepath):
    print("Caricamento dei dati medici")
    df = pd.read_csv(filepath)
    print(df.head())
    return df

def preprocess_data(df):
    print("Pre-processing dei dati: rimozione colonne inutili")
    df = df.drop(columns=[
        'Name', 'Doctor', 'Hospital', 'Insurance Provider',
        'Date of Admission', 'Discharge Date', 'Room Number'
    ])
    df = df.dropna()

    x = df.drop(columns=['Medical Condition'])
    y = df['Medical Condition']

    le = LabelEncoder()
    for col in x.select_dtypes(include='object'):
        x[col] = le.fit_transform(x[col])
    
    y = le.fit_transform(y)

    return x, y

def train_model(x_train, y_train):
    print("Addestramento del modello di classificazione (Random Forest)")
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    print("Valutazione del modello")
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuratezza del modello: {acc:.2f}")
    return acc

if __name__ == '__main__':
    df = load_healthcare('../dataset/healthcare_dataset.csv')
    x, y = preprocess_data(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = train_model(x_train, y_train)
    evaluate_model(model, x_test, y_test)


'''
preprocess_data
- rimuove le colonne inutili
- rimuove i valori mancanti
- divide le feature x e y
- codifica le colonne categoriche con LabelEncoder:
    - per ogni colonna non numerica, converte stringhe in numeri interi
    - lo fa anche per la colonna y

LabelEncoder trasforma le etichette (stringhe) in numeri interi
Male 1, Female 0
E' semplice, ma introduce un ordine fittizio, che può essere problematico se il modello lo interpreta
come un valore continuo
posso usare CounteVectorizer come nell'analisi del sentiment?
No, perché è pensato per testo libero

'''