# Classification de Genres Musicaux avec GTZAN

## Description
Ce projet vise à classifier automatiquement les genres musicaux en utilisant le dataset GTZAN. Ce dataset contient 1 000 morceaux répartis en 10 genres différents (rock, jazz, blues, reggae, etc.). L'objectif est d'extraire des caractéristiques spectrales et temporelles des fichiers audio et d'entraîner un modèle de machine learning ou de deep learning pour la classification.

## Données
- **Nom du dataset** : GTZAN Dataset
- **Format des fichiers** : WAV
- **Nombre de classes** : 10 genres musicaux
- **Lien du dataset** : [GTZAN sur Kaggle](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

## Approche
1. **Prétraitement des données**
   - Chargement des fichiers audio
   - Normalisation et segmentation si nécessaire
2. **Extraction de caractéristiques**
   - MFCCs (Mel-Frequency Cepstral Coefficients)
   - Chroma Features
   - Spectrogrammes
3. **Modélisation**
   - Modèles de machine learning (Random Forest, SVM, KNN, etc.)
   - Réseaux de neurones profonds (CNN, RNN, LSTM)
4. **Évaluation**
   - Matrice de confusion
   - Analyse des erreurs et des genres les plus difficiles à différencier

## Prérequis
- Python 3.x
- Bibliothèques nécessaires :
  ```bash
  pip install librosa pandas numpy matplotlib scikit-learn tensorflow keras
  ```

## Utilisation
1. **Cloner le dépôt**
   ```bash
   git clone <URL_du_repo>
   cd music-genre-classification
   ```
2. **Exécuter le script de classification**
   ```bash
   python main.py
   ```

## Résultats attendus
- Un modèle capable de classifier automatiquement les genres musicaux avec une précision acceptable.
- Une analyse des erreurs de classification pour améliorer les performances du modèle.

## Auteurs
- Maxens SOLDAN
- Baptiste RENAND 

## Licence
Ce projet est sous licence MIT.