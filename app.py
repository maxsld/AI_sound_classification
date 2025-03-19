import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Charger les données
df = pd.read_csv("MusicGenre/features_30_sec.csv")  # Remplace par ton fichier

# 2. Séparer les features et les labels
X = df.drop(columns=["filename", "label"])  # Supprimer le nom du fichier
y = df["label"]

# 3. Prétraitement : Normalisation des features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=176)

# 5. Entraîner le modèle Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=176)
clf.fit(X_train, y_train)

# 6. Prédictions et évaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
