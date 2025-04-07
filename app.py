import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns

# Charger les données
df = pd.read_csv("MusicGenre/features_30_sec.csv")

# Séparer les features et les labels
X = df.drop(columns=["filename", "label"])
y = df["label"]

# Encoder les labels
y_encoded = LabelEncoder().fit_transform(y)

# Normalisation des features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=176)

# Validation croisée k-fold
k = 5

# Fonction d'évaluation
def evaluate_model(model, name):
    scores = cross_val_score(model, X_train, y_train, cv=k)
    print(f"{name} - Validation croisée ({k}-fold) - Score moyen: {scores.mean():.2f}")
    return scores.mean()

# Fonction pour afficher la matrice de confusion
def plot_confusion_matrix(y_true, y_pred, model_name):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LabelEncoder().fit(y).classes_, 
                yticklabels=LabelEncoder().fit(y).classes_)
    plt.title(f"Matrice de Confusion pour {model_name}")
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités")
    plt.show()

# 1. Random Forest avec optimisation
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
clf_rf = GridSearchCV(RandomForestClassifier(random_state=176), param_grid_rf, cv=k, n_jobs=-1)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.best_estimator_.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
evaluate_model(clf_rf.best_estimator_, "Random Forest")
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")

# 2. k-NN avec optimisation
param_grid_knn = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
clf_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=k, n_jobs=-1)
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.best_estimator_.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
evaluate_model(clf_knn.best_estimator_, "KNN")
plot_confusion_matrix(y_test, y_pred_knn, "KNN")

# 3. Decision Tree avec optimisation
param_grid_dt = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
clf_dt = GridSearchCV(DecisionTreeClassifier(random_state=176), param_grid_dt, cv=k, n_jobs=-1)
clf_dt.fit(X_train, y_train)
y_pred_dt = clf_dt.best_estimator_.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
evaluate_model(clf_dt.best_estimator_, "Decision Tree")
plot_confusion_matrix(y_test, y_pred_dt, "Decision Tree")

# 4. MLP avec optimisation et corrections de convergence
param_grid_mlp = {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01], 'solver': ['adam', 'sgd', 'lbfgs']}
clf_mlp = GridSearchCV(MLPClassifier(max_iter=1500, random_state=176), param_grid_mlp, cv=k, n_jobs=-1)
clf_mlp.fit(X_train, y_train)
y_pred_mlp = clf_mlp.best_estimator_.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
evaluate_model(clf_mlp.best_estimator_, "MLP")
plot_confusion_matrix(y_test, y_pred_mlp, "MLP")

# 5. Minimal Distance Method (MDM)
clf_mdm = NearestCentroid()
clf_mdm.fit(X_train, y_train)
y_pred_mdm = clf_mdm.predict(X_test)
accuracy_mdm = accuracy_score(y_test, y_pred_mdm)
evaluate_model(clf_mdm, "MDM")
plot_confusion_matrix(y_test, y_pred_mdm, "MDM")

# Comparaison des modèles
models = ["Random Forest", "KNN", "Decision Tree", "MLP", "MDM"]
accuracies = [accuracy_rf, accuracy_knn, accuracy_dt, accuracy_mlp, accuracy_mdm]
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel("Modèles")
plt.ylabel("Précision")
plt.title("Comparaison des modèles de classification après optimisation")
plt.ylim(0.4, 1.0)
plt.show()
