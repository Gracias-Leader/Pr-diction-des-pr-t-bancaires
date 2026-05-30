# 🏦 Segmentation & Prédiction d'Éligibilité aux Prêts Bancaires

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-red)
![Methodology](https://img.shields.io/badge/Methodology-CRISP--DM-green)
![Status](https://img.shields.io/badge/Status-Complet-brightgreen)

---

## 📋 Contexte Métier

Les institutions bancaires font face à un double défi : **identifier les clients éligibles** à un prêt tout en **optimisant leur stratégie commerciale** par segment de clientèle.

Ce projet répond à ces deux besoins via une approche hybride :

- **Partie A — Segmentation (Non Supervisée)** : Regroupement des clients par profil de prêt via K-Means, pour cibler les offres commerciales (micro-crédits, prêts immobiliers, etc.)
- **Partie B — Prédiction d'Éligibilité (Supervisée)** : Automatisation de la décision d'octroi de prêt en temps réel, à partir des informations du formulaire de demande

---

## 📁 Structure du Projet

```
├── train_segment.ipynb     # Notebook principal (analyse complète)
├── train.csv               # Dataset d'entraînement
└── README.md
```

---

## 📊 Dataset

Le dataset contient **614 demandes de prêt** avec les variables suivantes :

| Variable | Type | Description |
|---|---|---|
| `Gender` | Catégorielle | Genre du demandeur |
| `Married` | Catégorielle | Statut marital |
| `Dependents` | Catégorielle | Nombre de personnes à charge |
| `Education` | Catégorielle | Niveau d'études |
| `Self_Employed` | Catégorielle | Travailleur indépendant |
| `ApplicantIncome` | Numérique | Revenu du demandeur |
| `CoapplicantIncome` | Numérique | Revenu du co-demandeur |
| `LoanAmount` | Numérique | Montant du prêt demandé |
| `Loan_Amount_Term` | Numérique | Durée du prêt |
| `Credit_History` | Binaire | Historique de crédit (1 = positif) |
| `Property_Area` | Catégorielle | Zone géographique |
| `Loan_Status` ⭐ | Cible | Éligibilité (Y/N) |

---

## 🔬 Méthodologie : CRISP-DM

```
1. Business Understanding  →  Définition des objectifs métier
2. Data Understanding      →  Exploration et description du dataset
3. Data Preparation        →  Nettoyage, imputation, encodage
4. Modeling                →  K-Means + 3 modèles de classification
5. Evaluation              →  Recall, F1-Score, AUC-ROC
6. Deployment              →  Sélection du meilleur modèle
```

---

## ⚙️ Pipeline Technique

### Prétraitement
- Suppression de `Loan_ID` (identifiant non informatif)
- **Imputation** : mode pour les variables catégorielles, moyenne pour les numériques
- **Encodage** : mapping binaire pour les variables à 2 modalités, `get_dummies` pour les variables multi-modalités

### Modèles entraînés

| Modèle | Gestion du déséquilibre |
|---|---|
| Régression Logistique | — |
| Random Forest | `class_weight="balanced"` |
| XGBoost | `scale_pos_weight` calculé |

Chaque modèle est encapsulé dans un **scikit-learn Pipeline** (StandardScaler + Estimateur).

### Optimisation
- **GridSearchCV** avec validation croisée (cv=5), métrique principale : **Recall**
- Sélection du meilleur modèle selon Recall / F1-Score / AUC-ROC sur le test set

---

## 📈 Résultats

Le choix du **Recall** comme métrique principale est justifié par le contexte bancaire : il est plus coûteux de **rejeter un client éligible** (manque à gagner) que d'approuver un dossier limite. Les résultats détaillés (matrices de confusion, classification reports, courbes ROC) sont disponibles dans le notebook.

---

## 🛠️ Installation & Exécution

### Prérequis

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy
```

### Lancer le notebook

```bash
jupyter notebook train_segment.ipynb
```

---

## 🧪 Tests Statistiques

Des tests **Mann-Whitney U** ont été appliqués sur l'ensemble des variables (après encodage) pour évaluer leur significativité vis-à-vis de la variable cible `Loan_Status`, avec un seuil α = 0.05.

---

## 👤 Auteur

**Bertin** — Data Science Practitioner  
🔗 [GitHub : Gracias-Leader](https://github.com/Gracias-Leader)
