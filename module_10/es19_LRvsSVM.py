# LOGISTIC REGRESSION VS SUPPORT VECTOR MACHINE
# 19) Sperimenta la differenza tra la regressione logistica e un modello SVM per un problema di classificazione binaria. Valuta le prestazioni di entrambi gli algoritmi su un dataset e confrontali.

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

def load_data():
    data = load_breast_cancer()
    x = data.data
    y = data.target
    print("Feature names:", data.feature_names)
    print("Target names:", data.target_names)
    return x, y

def preprocess(x):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled

def train_and_evaluate(x_train, x_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "SVM (Linear Kernel)": SVC(kernel="linear", probability=True)
    }

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1]

        print(f"\n{name}")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # ROC curve (RECEIVER OPERATING CHARACTERISTIC curve)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    x, y = load_data()
    x_scaled = preprocess(x)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3)

    train_and_evaluate(x_train, x_test, y_train, y_test)
