# ENSEMLE
# 20) Utilizza algoritmi di ensemble (di classificazione), come Random Forest e Gradient Boosting per migliorare
# le prestazioni del tuo modello di machine learning. Addestra modelli singoli e confrontali con il modello ensemble.

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve

def load_data():
    data = load_breast_cancer()
    x = data.data
    y = data.target
    return x, y

def preprocess(x):
    scaler = StandardScaler()
    return scaler.fit_transform(x)

def train_and_compare(x_train, x_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
    }

    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(x_test)[:, 1]
        else:
            y_proba = model.decision_function(x_test)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        print(f"\n{name}")
        print(f"Accuracy: {acc:.3f}")
        print(f"ROC AUC: {auc:.3f}")
        print(classification_report(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, y_proba) # ROC Curve
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    x, y = load_data()
    x_scaled = preprocess(x)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3)
    train_and_compare(x_train, x_test, y_train, y_test)
