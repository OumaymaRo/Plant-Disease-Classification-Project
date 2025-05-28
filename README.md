# 🌿 Plant Disease Classification Project

Ce projet a pour objectif de **classifier automatiquement les maladies des plantes** à partir d’images de feuilles en niveaux de gris.  
Il combine des **techniques de traitement d’image** classiques avec un **modèle de machine learning (Random Forest)**.  
L’application finale permet à l’utilisateur de téléverser une image via une interface web (Flask) et d’obtenir une prédiction instantanée.

---

## 📊 Données

Le dataset utilisé est issu de Kaggle :

🔗 [PlantVillage Dataset - Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

Il comprend plus de **50 000 images** de feuilles saines et malades, couvrant plusieurs types de cultures (tomates, pommes de terre, maïs, etc.).

---

## 🛠️ Pipeline de traitement

### 🖼️ Prétraitement d’image :
- Conversion en niveaux de gris
- Filtrage médian pour suppression du bruit
- Amélioration des contours (Unsharp Mask)
- Ajustement du contraste
- Redimensionnement à 256x256

### 🧠 Extraction des caractéristiques :
- Histogrammes de niveaux de gris
- HOG (Histogram of Oriented Gradients)
- Moyenne et écart-type d’intensité

### 🌳 Modélisation :
- Entraînement avec un classificateur **Random Forest**
- Sauvegarde du modèle avec `joblib`
- Prédiction avec affichage de la **classe (plante + maladie)** et **confiance du modèle**

---

## 🌐 Interface Web

L’application est développée avec **Flask**.  
Une interface simple permet à l'utilisateur de :

- Téléverser une image
- Visualiser l'image d'entrée et l'image traitée
- Recevoir le **résultat de la prédiction** avec taux de confiance

📍 Pour lancer l'app localement :

[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 👥 Auteurs

- Oumayma Rokhssi  
- Kawtar Elbhiri  
- Aya Slassi  
- Bouthaina Tricha  
- Ghita Slaoui Hasnaoui  

