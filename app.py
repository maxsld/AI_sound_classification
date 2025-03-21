import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt

# 1. Charger les données
df = pd.read_csv("MusicGenre/features_30_sec.csv")  # Remplace par ton fichier

# 2. Séparer les features et les labels
X = df.drop(columns=["filename", "label"])  # Supprimer le nom du fichier
y = df["label"]

# 3. Encoder les labels de genre en valeurs numériques
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Prétraitement : Normalisation des features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=176)

# --- 1. Entraîner le modèle Random Forest ---
# 6. Entraîner le modèle Random Forest
clf_rf = RandomForestClassifier(n_estimators=100, random_state=176)
clf_rf.fit(X_train, y_train)

# 7. Prédictions et évaluation Random Forest
y_pred_rf = clf_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Random Forest - Accuracy: {accuracy_rf:.2f}")
print("Random Forest - Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# --- 2. Entraîner le modèle KNN ---
# 8. Entraîner le modèle KNN
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(X_train, y_train)

# 9. Prédictions et évaluation KNN
y_pred_knn = clf_knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"KNN - Accuracy: {accuracy_knn:.2f}")
print("KNN - Classification Report:\n", classification_report(y_test, y_pred_knn))
print("KNN - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

# --- 3. Entraîner le modèle Minimal Distance Method (MDM) ---
# 10. Entraîner le modèle MDM (NearestCentroid)
clf_mdm = NearestCentroid()
clf_mdm.fit(X_train, y_train)

# 11. Prédictions et évaluation MDM
y_pred_mdm = clf_mdm.predict(X_test)
accuracy_mdm = accuracy_score(y_test, y_pred_mdm)

print(f"MDM (Minimal Distance Method) - Accuracy: {accuracy_mdm:.2f}")
print("MDM - Classification Report:\n", classification_report(y_test, y_pred_mdm))
print("MDM - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mdm))

